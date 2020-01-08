# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import datetime
import sys
from path import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

import matplotlib.pyplot as plt
from utils.official import *
from utils.img_process import tensor2array,tensor2array2
from kitti_utils import *
from layers import *

from datasets import KITTIRAWDataset
from datasets import KITTIOdomDataset
from datasets import KITTIDepthDataset
from datasets import MCDataset
import networks
from utils.logger import TermLogger,AverageMeter
from SoftHistogram2D.soft_hist import SoftHistogram2D_H

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.checkpoints_path = Path(self.opt.log_dir)/datetime.datetime.now().strftime("%m-%d-%H:%M")
        #save model and events


        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}#dict
        self.parameters_to_train = []#list

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        #self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])



        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        #if self.opt.pose_model_type == "separate_resnet":
        self.models["pose_encoder"] = networks.ResnetEncoder(
                                        self.opt.num_layers,
                                        self.opt.weights_init == "pretrained",
                                        num_input_images=self.num_pose_frames)

        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)



        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())



        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\t  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\t  ", self.opt.log_dir)
        print("Training is using:\t  ", self.device)

        # datasets setting
        datasets_dict = {"kitti": KITTIRAWDataset,
                         "kitti_odom": KITTIOdomDataset,
                         "mc":MCDataset}
        self.dataset = datasets_dict[self.opt.dataset]#选择建立哪个类，这里kitti，返回构造函数句柄

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        #train loader
        train_dataset = self.dataset(#KITTIRAWData
            self.opt.data_path,
            train_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=True,
            img_ext=img_ext)
        self.train_loader = DataLoader(#train_datasets:KITTIRAWDataset
            dataset=train_dataset,
            batch_size= self.opt.batch_size,
            shuffle= False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        #val loader
        val_dataset = self.dataset(
            self.opt.data_path,
            val_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=False,
            img_ext=img_ext)

        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(self.checkpoints_path/mode)

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\t  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

        #custom

        self.logger = TermLogger(n_epochs=self.opt.num_epochs,
                            train_size=len(self.train_loader),
                            valid_size=len(self.val_loader))
        self.logger.reset_epoch_bar()

        self.metrics = {}


        self.histc_loss = [SoftHistogram2D_H(self.device,bins=int(self.opt.height/2),min=0,max=1,sigma=512,b=1,c=1,h=self.opt.height,w = self.opt.width),
                           SoftHistogram2D_H(self.device, bins=int(self.opt.height/4), min=0, max=1, sigma=512, b=1, c=1,h=int(self.opt.height/2), w=int(self.opt.width/2)),
                           SoftHistogram2D_H(self.device, bins=int(self.opt.height/8), min=0, max=1, sigma=512, b=1, c=1,h=int(self.opt.height/4), w=int(self.opt.width/4)),
                           SoftHistogram2D_H(self.device, bins=int(self.opt.height/16), min=0, max=1, sigma=512, b=1, c=1,h=int(self.opt.height/8), w=int(self.opt.width/8))
                            ]



    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    #1. forward pass1, more like core
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)


        #1. depth output

            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            # 关于output是由depth model来构型的
            # outputs only have disp 0,1,2,3

        if self.opt.geometry_loss_weights==0:
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)
            outputs[("disp", 0, 0)] = outputs[("disp", 0)]
            outputs[("disp", 0, 1)] = outputs[("disp", 1)]
            outputs[("disp", 0, 2)] = outputs[("disp", 2)]
            outputs[("disp", 0, 3)] = outputs[("disp", 3)]
        else:
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            features_m1 = self.models["encoder"](inputs["color_aug", -1, 0])
            features_1 = self.models["encoder"](inputs["color_aug", 1, 0])

            outputs_m1 = self.models["depth"](features_m1)
            outputs_0 = self.models["depth"](features)
            outputs_1 = self.models["depth"](features_1)
            outputs={}
            outputs[("disp",-1,0)] = outputs_m1[("disp",0)]
            #outputs[("disp",-1,1)] = outputs_m1[("disp",1)]
            #outputs[("disp",-1,2)] = outputs_m1[("disp",2)]
            #outputs[("disp",-1,3)] = outputs_m1[("disp",3)]


            outputs[("disp",0,0)] = outputs_0[("disp",0)]
            outputs[("disp",0,1)] = outputs_0[("disp",1)]
            outputs[("disp",0,2)] = outputs_0[("disp",2)]
            outputs[("disp",0,3)] = outputs_0[("disp",3)]


            outputs[("disp",1,0)] = outputs_1[("disp",0)]
            #outputs[("disp",1,1)] = outputs_1[("disp",1)]
            #outputs[("disp",1,2)] = outputs_1[("disp",2)]
            #outputs[("disp",1,3)] = outputs_1[("disp",3)]


        #2. mask

        #3. pose
        outputs.update(self.predict_poses(inputs, features))        #outputs get 3 new values


        #4.
        self.generate_images_pred(inputs, outputs)#outputs get
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    #2. called by 1
    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        write outputs
        """
        outputs = {}

        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.

        # select what features the pose network takes as input

        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}



        for f_i in self.opt.frame_ids[1:]:

            # To maintain ordering we always pass frames in temporal order

            #map concat
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]


            #encoder map
            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

            #decoder
            axisangle, translation = self.models["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))


        return outputs



    #3.
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary as color_identity.
        """
        for scale in self.opt.scales:
            # get depth
            if self.opt.geometry_loss_weights!=0:
                for frame_id in self.opt.frame_ids:
                    disp_key = ("disp",frame_id,scale)
                    depth_key = ("depth",frame_id,scale)
                    if disp_key in outputs.keys():
                        disp = outputs[disp_key]
                        disp = F.interpolate(
                            disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                        source_scale = 0

                        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

                        outputs[depth_key] = depth

            else:
                disp = outputs[("disp", 0,scale)]

                disp = F.interpolate(
                        disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

                outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):


                T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175


                cam_points = self.backproject_depth[source_scale](depth,
                                                                inputs[("inv_K", source_scale)])# D@K_inv
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)# K@D@K_inv


                outputs[("sample", frame_id, scale)] = pix_coords#rigid_flow

                outputs[("color", frame_id, scale)]= F.grid_sample(inputs[("color", frame_id, source_scale)],
                                                                    outputs[("sample", frame_id, scale)],
                                                                    padding_mode="border")
                #output"color" 就是i-warped

                #add a depth warp

                if self.opt.geometry_loss_weights!=0:
                    outputs[("depth_warp",frame_id,scale)] = F.grid_sample(outputs[("depth",frame_id,source_scale)],
                                                                       outputs[("sample", frame_id, scale)],
                                                                       padding_mode="border")


                outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

    #4.
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)#[b,1,h,w]


        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss#[b,1,h,w]

    def compute_geometry_loss(self,pred_depth,depth_warp):
        diff_depth = ((pred_depth - depth_warp).abs() /
                      (pred_depth + depth_warp).abs()).clamp(0, 1)


        return diff_depth
    #5.

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []


            source_scale = 0

            disp = outputs[("disp", 0,scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)


            identity_reprojection_loss = identity_reprojection_losses




            reprojection_loss = reprojection_losses

            # add random numbers to break ties# 花书p149 向输入添加方差极小的噪声等价于 对权重施加范数惩罚
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)


            to_optimise, idxs = torch.min(combined, dim=1)

            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_losses_1(self, inputs, outputs):
        """
        softmin and hardmin, first works better!
        L_p*MS_p
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_g=0
            loss_p=0

            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0,scale)]
            color = inputs[("color", 0, scale)]#??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:#前一帧,后一帧,两次计算
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)# list b,1,h,w  to b2hw






            #auto masking Enable/Disable
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))# total [b,2,h,w], 2 means pred frame and post frame
                #直接用相邻帧做损失, loss很小的地方说明后三种静止
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)#list2batch


            identity_reprojection_loss = identity_reprojection_losses
            #avg reprojection loss Enable/Disable
            reprojection_loss = reprojection_losses


            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            #
            soft_mask_p = 1.-torch.softmax(combined,dim=1)#softmin, 越小权值越大
            to_optimise = combined*soft_mask_p

            idxs = torch.argmax(soft_mask_p,dim=1)#mask大的说明重点优化

            outputs["identity_selection/{}".format(scale)] = idxs.float()  # 只读






            #idxs standsfor which layer of 4 is minimum of tensor in dimension 1

            # 将合成图像,loss更小的位置,置一, 这些地方属于前三种匹配;
            # 如果实值图像连续帧匹配loss更小, 说明后三种静止, 这些像素不处理, 置零
            # (关于为啥静止的不要， 主要是如果在一个地方估计出两次（静止）一样的错误， 会由于损失过小而判断这次估计非常精准)
            # 只是输出， 该地址的数据并不参与计算
            # 只记录来自2,3 即reprojection loss的地方,这写地方为min 且被选中意味着前3种匹配(不包括out of view与occluded pixels),此作为identity_mask
            # 0,1部分代表identity_reprojection loss, 通过两张相邻帧算出,如果min来自这里意味着后三种静止



            # \hat{I}_{t-1}, I_t, \hat{I}_{t+1} --> reprojection_loss
            # I_{t-1}, I_t, I_{t+1} --> identity_reprojection_loss


            # 0,1,2,3; 0,1是实值图像, 2,3是合成图像的索引
            #bhw


            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_p+= to_optimise.mean()
            losses["loss_p/{}".format(scale)] =  loss_p




            if self.opt.disparity_smoothness !=0:
                smooth_loss = get_smooth_loss(norm_disp, color)#b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s=0


            total_loss += loss_p+loss_s

        total_loss /= self.num_scales
        losses["loss"] = total_loss#mean of 4 scales
        return losses

    def compute_losses_2(self, inputs, outputs):
        """using geometry computed mask
        Lp * Mhg = -
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_g = 0
            loss_p = 0

            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0, scale)]
            color = inputs[("color", 0, scale)]  # ??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:  # 前一帧,后一帧,两次计算
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)  # list b,1,h,w  to b2hw

            # nips 2019
            if self.opt.geometry_loss_weights != 0:
                for frame_id in self.opt.frame_ids[1:]:
                    depth_warp = outputs[("depth_warp", frame_id, scale)]
                    geometry_losses.append(self.compute_geometry_loss(depth_warp, pred_depth))
                geometry_losses = torch.cat(geometry_losses, 1)

            # auto masking Enable/Disable
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))  # total [b,2,h,w], 2 means pred frame and post frame
                # 直接用相邻帧做损失, loss很小的地方说明后三种静止
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)  # list2batch

            identity_reprojection_loss = identity_reprojection_losses
            # avg reprojection loss Enable/Disable
            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)



            # idxs standsfor which layer of 4 is minimum of tensor in dimension 1

            # 将合成图像,loss更小的位置,置一, 这些地方属于前三种匹配;
            # 如果实值图像连续帧匹配loss更小, 说明后三种静止, 这些像素不处理, 置零
            # (关于为啥静止的不要， 主要是如果在一个地方估计出两次（静止）一样的错误， 会由于损失过小而判断这次估计非常精准)
            # 只是输出， 该地址的数据并不参与计算
            # 只记录来自2,3 即reprojection loss的地方,这写地方为min 且被选中意味着前3种匹配(不包括out of view与occluded pixels),此作为identity_mask
            # 0,1部分代表identity_reprojection loss, 通过两张相邻帧算出,如果min来自这里意味着后三种静止

            # \hat{I}_{t-1}, I_t, \hat{I}_{t+1} --> reprojection_loss
            # I_{t-1}, I_t, I_{t+1} --> identity_reprojection_loss

            if self.opt.geometry_loss_weights != 0:
                combined_g = torch.cat((identity_reprojection_loss, geometry_losses), dim=1)
                idxs = torch.argmin(combined_g, dim=1)  # idxs_g应该和
            else:
                idxs = torch.argmin(combined,dim=1)
            outputs["identity_selection/{}".format(scale)] = (idxs > 1).float()  # 只读

            to_optimise = torch.gather(combined,index=idxs[:, None, ...],dim=1)
            # 0,1,2,3; 0,1是实值图像, 2,3是合成图像的索引
            # bhw

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_p += to_optimise.mean()
            losses["loss_p/{}".format(scale)] = loss_p



            if self.opt.disparity_smoothness != 0:
                smooth_loss = get_smooth_loss(norm_disp, color)  # b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s = 0

            if self.opt.histc_weights != 0:
                histc_loss = get_hitc_loss(self.histc_loss[scale], norm_disp)
                loss_h = self.opt.histc_weights * histc_loss / (2 ** scale)
                losses["loss_h/{}".format(scale)] = loss_h
            else:
                loss_h = 0

            total_loss += loss_p + loss_s + loss_h + loss_g

        total_loss /= self.num_scales
        losses["loss"] = total_loss  # mean of 4 scales
        return losses
    def compute_losses_3(self, inputs, outputs):
        """
        L_p*MS_p.mean() + Lg*MS_g.mean()
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_g=0
            loss_p=0

            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0,scale)]
            color = inputs[("color", 0, scale)]#??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:#前一帧,后一帧,两次计算
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)# list b,1,h,w  to b2hw


            #nips 2019
            if self.opt.geometry_loss_weights!=0:
                for frame_id in self.opt.frame_ids[1:]:
                    depth_warp = outputs[("depth_warp", frame_id, scale)]
                    geometry_losses.append(self.compute_geometry_loss(depth_warp,pred_depth))
                geometry_losses = torch.cat(geometry_losses, 1)



            #auto masking Enable/Disable
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))# total [b,2,h,w], 2 means pred frame and post frame
                #直接用相邻帧做损失, loss很小的地方说明后三种静止
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)#list2batch


            identity_reprojection_loss = identity_reprojection_losses
            #avg reprojection loss Enable/Disable
            reprojection_loss = reprojection_losses


            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)


            if self.opt.geometry_loss_weights!=0:
                combined_g = torch.cat((identity_reprojection_loss, geometry_losses), dim=1)
                to_optimise_g, idxs_g = torch.min(combined_g,dim=1)#idxs_g应该和


            soft_idxs = 1.-torch.softmax(combined,dim=1)#softmin
            to_optimise = combined*soft_idxs

            idxs = torch.argmax(soft_idxs,dim=1)


            soft_idxs_g = 1.-torch.softmax(combined_g,dim=1)
            to_optimise_g = combined_g *soft_idxs_g
            idxs_g = torch.argmax(soft_idxs_g,dim=1)

            outputs["identity_selection/{}".format(scale)] = idxs.float()  # 只读
            outputs["identity_selection_g/{}".format(scale)] = idxs_g.float()  # 只读





            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_p+= to_optimise.mean()
            losses["loss_p/{}".format(scale)] =  loss_p

            loss_g += to_optimise_g.mean()
            losses["loss_g/{}".format(scale)] = loss_g



            if self.opt.disparity_smoothness !=0 and scale==0:
                smooth_loss = get_smooth_loss(norm_disp, color)#b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s=0



            total_loss += loss_p+loss_s+loss_g

        total_loss /= self.num_scales
        losses["loss"] = total_loss#mean of 4 scales
        return losses
    def compute_losses_4(self, inputs, outputs):
        """
        （L_p*Lg）（MS_p × MS_g）
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_pg=0

            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0,scale)]
            color = inputs[("color", 0, scale)]#??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:#前一帧,后一帧,两次计算
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)# list b,1,h,w  to b2hw


            #nips 2019
            if self.opt.geometry_loss_weights!=0:
                for frame_id in self.opt.frame_ids[1:]:
                    depth_warp = outputs[("depth_warp", frame_id, scale)]
                    geometry_losses.append(self.compute_geometry_loss(depth_warp,pred_depth))
                geometry_losses = torch.cat(geometry_losses, 1)



            #auto masking Enable/Disable
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))# total [b,2,h,w], 2 means pred frame and post frame
                #直接用相邻帧做损失, loss很小的地方说明后三种静止
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)#list2batch


            identity_reprojection_loss = identity_reprojection_losses
            #avg reprojection loss Enable/Disable
            reprojection_loss = reprojection_losses



            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)


            combined_g = torch.cat((identity_reprojection_loss, geometry_losses), dim=1)
            #to_optimise_g, idxs_g = torch.min(combined_g,dim=1)#idxs_g应该和

            combined_pg = combined*combined_g

            #if self.opt.softmin:
            soft_idxs_pg = 1.-torch.softmax(combined_pg,dim=1)#softmin
            to_optimise_pg = combined_pg*soft_idxs_pg#

            idxs_pg = torch.argmin(soft_idxs_pg,dim=1)



            outputs["identity_selection/{}".format(scale)] = idxs_pg.float()  # 只读







            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_pg+= to_optimise_pg.mean()
            losses["loss_pg/{}".format(scale)] =  loss_pg




            if self.opt.disparity_smoothness !=0 and scale==0:#smooth就只放一个scale 0 的就行了
                smooth_loss = get_smooth_loss(norm_disp, color)#b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s=0



            total_loss += loss_pg+loss_s

        total_loss /= self.num_scales
        losses["loss"] = total_loss#mean of 4 scales
        return losses

    def compute_losses_5(self, inputs, outputs):
        """
        (Lp MSpMsg +Lp MSpMsg).mean()

        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_pg=0
            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0,scale)]
            color = inputs[("color", 0, scale)]#??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:#前一帧,后一帧,两次计算
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)# list b,1,h,w  to b2hw


            #nips 2019
            if self.opt.geometry_loss_weights!=0:
                for frame_id in self.opt.frame_ids[1:]:
                    depth_warp = outputs[("depth_warp", frame_id, scale)]
                    geometry_losses.append(self.compute_geometry_loss(depth_warp,pred_depth))
                geometry_losses = torch.cat(geometry_losses, 1)



            #auto masking Enable/Disable
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))# total [b,2,h,w], 2 means pred frame and post frame
                #直接用相邻帧做损失, loss很小的地方说明后三种静止
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)#list2batch


            identity_reprojection_loss = identity_reprojection_losses
            #avg reprojection loss Enable/Disable
            reprojection_loss = reprojection_losses



            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            combined_p = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)


            combined_g = torch.cat((identity_reprojection_loss, geometry_losses), dim=1)
            #to_optimise_g, idxs_g = torch.min(combined_g,dim=1)#idxs_g应该和

            #combined_pg = combined*combined_g

            #if self.opt.softmin:
            soft_idxs_p = 1.-torch.softmax(combined_p,dim=1)#softmin
            #to_optimise_p = combined_p*soft_idxs_p#
            #idxs_p = torch.argmax(soft_idxs_p,dim=1)


            soft_idxs_g = 1. - torch.softmax(combined_p, dim=1)  # softmin
            #to_optimise_g = combined_g * soft_idxs_g  #
            #idxs_g = torch.argmax(soft_idxs_g,dim=1)

            mask = soft_idxs_p*soft_idxs_g
            to_optimise_pg = combined_p*mask + combined_g * mask

            idxs = torch.argmax(mask,dim=1)
            outputs["identity_selection/{}".format(scale)] = idxs.float()  # 只读
            #outputs["identity_selection_g/{}".format(scale)] = idxs_g.float()  # 只读








            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_pg+= to_optimise_pg.mean()

            losses["loss_pg/{}".format(scale)] =  loss_pg





            if self.opt.disparity_smoothness !=0 and scale==0:#smooth就只放一个scale 0 的就行了
                smooth_loss = get_smooth_loss(norm_disp, color)#b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s=0



            total_loss += loss_pg+loss_s

        total_loss /= self.num_scales
        losses["loss"] = total_loss#mean of 4 scales
        return losses
    #6.
    def compute_losses_6(self, inputs, outputs):
        """
        (LpMSg +LgMSg).mean()
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_pg=0
            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0,scale)]
            color = inputs[("color", 0, scale)]#??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:#前一帧,后一帧,两次计算
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)# list b,1,h,w  to b2hw


            #nips 2019
            if self.opt.geometry_loss_weights!=0:
                for frame_id in self.opt.frame_ids[1:]:
                    depth_warp = outputs[("depth_warp", frame_id, scale)]
                    geometry_losses.append(self.compute_geometry_loss(depth_warp,pred_depth))
                geometry_losses = torch.cat(geometry_losses, 1)



            #auto masking Enable/Disable
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))# total [b,2,h,w], 2 means pred frame and post frame
                #直接用相邻帧做损失, loss很小的地方说明后三种静止
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)#list2batch


            identity_reprojection_loss = identity_reprojection_losses
            #avg reprojection loss Enable/Disable
            reprojection_loss = reprojection_losses



            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            combined_p = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            combined_g = torch.cat((identity_reprojection_loss, geometry_losses), dim=1)


            #to_optimise_g, idxs_g = torch.min(combined_g,dim=1)#idxs_g应该和

            #combined_pg = combined*combined_g

            #if self.opt.softmin:
            mask_p = 1.-torch.softmax(combined_p,dim=1)#softmin
            mask_g = 1.-torch.softmax(combined_g,dim=1)#softmin



            to_optimise_pg = combined_p*mask_p + combined_g * mask_g

            idxs_p = torch.argmax(mask_p,dim=1)
            idxs_g = torch.argmax(mask_g,dim=1)

            outputs["identity_selection/{}".format(scale)] = idxs_p.float()  # 只读
            outputs["identity_selection_g/{}".format(scale)] = idxs_g.float()  # 只读








            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_pg+= to_optimise_pg.mean()

            losses["loss_pg/{}".format(scale)] =  loss_pg





            if self.opt.disparity_smoothness !=0 and scale==0:#smooth就只放一个scale 0 的就行了
                smooth_loss = get_smooth_loss(norm_disp, color)#b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s=0



            total_loss += loss_pg+loss_s

        total_loss /= self.num_scales
        losses["loss"] = total_loss#mean of 4 scales
        return losses

    def compute_losses_7(self, inputs, outputs):
        """
            LpMSpg + LgMSpg
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_pg = 0
            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0, scale)]
            color = inputs[("color", 0, scale)]  # ??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:  # 前一帧,后一帧,两次计算
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)  # list b,1,h,w  to b2hw

            # nips 2019
            if self.opt.geometry_loss_weights != 0:
                for frame_id in self.opt.frame_ids[1:]:
                    depth_warp = outputs[("depth_warp", frame_id, scale)]
                    geometry_losses.append(self.compute_geometry_loss(depth_warp, pred_depth))
                geometry_losses = torch.cat(geometry_losses, 1)

            # auto masking Enable/Disable
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))  # total [b,2,h,w], 2 means pred frame and post frame
                # 直接用相邻帧做损失, loss很小的地方说明后三种静止
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)  # list2batch

            identity_reprojection_loss = identity_reprojection_losses
            # avg reprojection loss Enable/Disable
            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            combined_p = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            combined_g = torch.cat((identity_reprojection_loss, geometry_losses), dim=1)
            # to_optimise_g, idxs_g = torch.min(combined_g,dim=1)#idxs_g应该和

            # combined_pg = combined*combined_g

            # if self.opt.softmin:
            mask = 1. - torch.softmax(combined_p * combined_g, dim=1)  # softmin

            to_optimise_pg = combined_p * mask + combined_g * mask

            idxs = torch.argmax(mask, dim=1)
            outputs["identity_selection/{}".format(scale)] = idxs.float()  # 只读
            # outputs["identity_selection_g/{}".format(scale)] = idxs_g.float()  # 只读

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_pg += to_optimise_pg.mean()

            losses["loss_pg/{}".format(scale)] = loss_pg

            if self.opt.disparity_smoothness != 0 and scale == 0:  # smooth就只放一个scale 0 的就行了
                smooth_loss = get_smooth_loss(norm_disp, color)  # b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s = 0

            total_loss += loss_pg + loss_s

        total_loss /= self.num_scales
        losses["loss"] = total_loss  # mean of 4 scales
        return losses
    def compute_losses_8(self, inputs, outputs):
        """
        (LpMSg +0.5 LgMSg).mean()
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_pg=0
            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0,scale)]
            color = inputs[("color", 0, scale)]#??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:#前一帧,后一帧,两次计算
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)# list b,1,h,w  to b2hw


            #nips 2019
            if self.opt.geometry_loss_weights!=0:
                for frame_id in self.opt.frame_ids[1:]:
                    depth_warp = outputs[("depth_warp", frame_id, scale)]
                    geometry_losses.append(self.compute_geometry_loss(depth_warp,pred_depth))
                geometry_losses = torch.cat(geometry_losses, 1)



            #auto masking Enable/Disable
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))# total [b,2,h,w], 2 means pred frame and post frame
                #直接用相邻帧做损失, loss很小的地方说明后三种静止
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)#list2batch


            identity_reprojection_loss = identity_reprojection_losses
            #avg reprojection loss Enable/Disable
            reprojection_loss = reprojection_losses



            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            combined_p = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            combined_g = torch.cat((identity_reprojection_loss, geometry_losses), dim=1)


            #to_optimise_g, idxs_g = torch.min(combined_g,dim=1)#idxs_g应该和

            #combined_pg = combined*combined_g

            #if self.opt.softmin:
            mask_p = 1.-torch.softmax(combined_p,dim=1)#softmin
            mask_g = 1.-torch.softmax(combined_g,dim=1)#softmin



            to_optimise_pg = combined_p*mask_p + 0.5*combined_g * mask_g

            idxs_p = torch.argmax(mask_p,dim=1)
            idxs_g = torch.argmax(mask_g,dim=1)

            outputs["identity_selection/{}".format(scale)] = idxs_p.float()  # 只读
            outputs["identity_selection_g/{}".format(scale)] = idxs_g.float()  # 只读








            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_pg+= to_optimise_pg.mean()

            losses["loss_pg/{}".format(scale)] =  loss_pg





            if self.opt.disparity_smoothness !=0 and scale==0:#smooth就只放一个scale 0 的就行了
                smooth_loss = get_smooth_loss(norm_disp, color)#b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s=0



            total_loss += loss_pg+loss_s

        total_loss /= self.num_scales
        losses["loss"] = total_loss#mean of 4 scales
        return losses

    def compute_losses_9(self, inputs, outputs):
        """
        (LpMSg +0.5 LgMSg).mean()
        lg更改为与 实值图像序列无关
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_pg=0
            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0,scale)]
            color = inputs[("color", 0, scale)]#??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:#前一帧,后一帧,两次计算
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)# list b,1,h,w  to b2hw


            #nips 2019
            if self.opt.geometry_loss_weights!=0:
                for frame_id in self.opt.frame_ids[1:]:
                    depth_warp = outputs[("depth_warp", frame_id, scale)]
                    geometry_losses.append(self.compute_geometry_loss(depth_warp,pred_depth))
                geometry_losses = torch.cat(geometry_losses, 1)



            #auto masking Enable/Disable
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))# total [b,2,h,w], 2 means pred frame and post frame
                #直接用相邻帧做损失, loss很小的地方说明后三种静止
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)#list2batch


            identity_reprojection_loss = identity_reprojection_losses
            #avg reprojection loss Enable/Disable
            reprojection_loss = reprojection_losses



            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            combined_p = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)#b4hw
            combined_g = geometry_losses #b2hw


            #to_optimise_g, idxs_g = torch.min(combined_g,dim=1)#idxs_g应该和

            #combined_pg = combined*combined_g

            #if self.opt.softmin:
            mask_p = 1.-torch.softmax(combined_p,dim=1)##b4hw
            mask_g = 1.-torch.softmax(combined_g,dim=1)#softmin



            to_optimise_pg = (combined_p*mask_p).mean(dim=1) + (0.5*combined_g * mask_g).mean(dim=1)

            idxs_p = torch.argmax(mask_p,dim=1)
            idxs_g = torch.argmax(mask_g,dim=1)

            outputs["identity_selection/{}".format(scale)] = idxs_p.float()  # 只读
            outputs["identity_selection_g/{}".format(scale)] = idxs_g.float()  # 只读








            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_pg+= to_optimise_pg.mean()

            losses["loss_pg/{}".format(scale)] =  loss_pg





            if self.opt.disparity_smoothness !=0 and scale==0:#smooth就只放一个scale 0 的就行了
                smooth_loss = get_smooth_loss(norm_disp, color)#b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s=0



            total_loss += loss_pg+loss_s

        total_loss /= self.num_scales
        losses["loss"] = total_loss#mean of 4 scales
        return losses

    def compute_losses_10(self, inputs, outputs):
        """
        (LpMSg +0.5 Lg*MSg).mean()
        lg更改为与 实值图像序列无关
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_pg=0
            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0,scale)]
            color = inputs[("color", 0, scale)]#??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:#前一帧,后一帧,两次计算
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)# list b,1,h,w  to b2hw


            #nips 2019
            if self.opt.geometry_loss_weights!=0:
                for frame_id in self.opt.frame_ids[1:]:
                    depth_warp = outputs[("depth_warp", frame_id, scale)]
                    geometry_losses.append(self.compute_geometry_loss(depth_warp,pred_depth))
                geometry_losses = torch.cat(geometry_losses, 1)



            #auto masking Enable/Disable
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))# total [b,2,h,w], 2 means pred frame and post frame
                #直接用相邻帧做损失, loss很小的地方说明后三种静止
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)#list2batch


            identity_reprojection_loss = identity_reprojection_losses
            #avg reprojection loss Enable/Disable
            reprojection_loss = reprojection_losses



            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            combined_p = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)#b4hw
            combined_g = geometry_losses #b2hw


            #to_optimise_g, idxs_g = torch.min(combined_g,dim=1)#idxs_g应该和

            #combined_pg = combined*combined_g

            #if self.opt.softmin:
            mask_p = 1.-torch.softmax(combined_p,dim=1)##b4hw
            mask_g = 1.-torch.softmax(combined_g,dim=1)#softmin



            to_optimise_pg = (combined_p*mask_p).mean(dim=1) + (0.2*combined_g * mask_g).mean(dim=1)

            idxs_p = torch.argmax(mask_p,dim=1)
            idxs_g = torch.argmax(mask_g,dim=1)

            outputs["identity_selection/{}".format(scale)] = idxs_p.float()  # 只读
            outputs["identity_selection_g/{}".format(scale)] = idxs_g.float()  # 只读








            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_pg+= to_optimise_pg.mean()

            losses["loss_pg/{}".format(scale)] =  loss_pg





            if self.opt.disparity_smoothness !=0 and scale==0:#smooth就只放一个scale 0 的就行了
                smooth_loss = get_smooth_loss(norm_disp, color)#b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s=0



            total_loss += loss_pg+loss_s

        total_loss /= self.num_scales
        losses["loss"] = total_loss#mean of 4 scales
        return losses

    def compute_losses_11(self, inputs, outputs):
        """
       ((Lp+Lg)MSp).mean
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_pg=0
            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0,scale)]
            color = inputs[("color", 0, scale)]#??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:#前一帧,后一帧,两次计算
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)# list b,1,h,w  to b2hw


            #nips 2019
            if self.opt.geometry_loss_weights!=0:
                for frame_id in self.opt.frame_ids[1:]:
                    depth_warp = outputs[("depth_warp", frame_id, scale)]
                    geometry_losses.append(self.compute_geometry_loss(depth_warp,pred_depth))
                geometry_losses = torch.cat(geometry_losses, 1)



            #auto masking Enable/Disable
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))# total [b,2,h,w], 2 means pred frame and post frame
                #直接用相邻帧做损失, loss很小的地方说明后三种静止
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)#list2batch


            identity_reprojection_loss = identity_reprojection_losses
            #avg reprojection loss Enable/Disable
            reprojection_loss = reprojection_losses



            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            combined_p = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)#b4hw
            combined_g = torch.cat((identity_reprojection_loss,geometry_losses),dim=1) #b2hw


            #to_optimise_g, idxs_g = torch.min(combined_g,dim=1)#idxs_g应该和

            #combined_pg = combined*combined_g

            #if self.opt.softmin:
            mask_p = 1.-torch.softmax(combined_p,dim=1)##b4hw
            mask_g = 1.-torch.softmax(combined_g,dim=1)#softmin



            to_optimise_pg = (combined_p+combined_g)*mask_p

            idxs_p = torch.argmax(mask_p,dim=1)
            #idxs_g = torch.argmax(mask_g,dim=1)

            outputs["identity_selection/{}".format(scale)] = idxs_p.float()  # 只读
            #outputs["identity_selection_g/{}".format(scale)] = idxs_g.float()  # 只读








            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_pg+= to_optimise_pg.mean()

            losses["loss_pg/{}".format(scale)] =  loss_pg





            if self.opt.disparity_smoothness !=0 and scale==0:#smooth就只放一个scale 0 的就行了
                smooth_loss = get_smooth_loss(norm_disp, color)#b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s=0



            total_loss += loss_pg+loss_s

        total_loss /= self.num_scales
        losses["loss"] = total_loss#mean of 4 scales
        return losses

    def compute_losses_12(self, inputs, outputs):
        """
       (LpgMSpg).mean
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_pg = 0
            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0, scale)]
            color = inputs[("color", 0, scale)]  # ??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:  # 前一帧,后一帧,两次计算
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)  # list b,1,h,w  to b2hw

            # nips 2019
            if self.opt.geometry_loss_weights != 0:
                for frame_id in self.opt.frame_ids[1:]:
                    depth_warp = outputs[("depth_warp", frame_id, scale)]
                    geometry_losses.append(self.compute_geometry_loss(depth_warp, pred_depth))
                geometry_losses = torch.cat(geometry_losses, 1)

            # auto masking Enable/Disable
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))  # total [b,2,h,w], 2 means pred frame and post frame
                # 直接用相邻帧做损失, loss很小的地方说明后三种静止
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)  # list2batch

            identity_reprojection_loss = identity_reprojection_losses
            # avg reprojection loss Enable/Disable
            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            combined_pg = torch.cat((identity_reprojection_loss, reprojection_loss,geometry_losses), dim=1)  # b4hw,可能geometry loss概率更大， 所以更模糊

            # to_optimise_g, idxs_g = torch.min(combined_g,dim=1)#idxs_g应该和

            # combined_pg = combined*combined_g

            # if self.opt.softmin:
            mask_pg = 1. - torch.softmax(combined_pg, dim=1)  ##b4hw

            to_optimise_pg = combined_pg * mask_pg

            idxs_pg = torch.argmax(mask_pg, dim=1)

            outputs["identity_selection/{}".format(scale)] = idxs_pg.float()  # 只读
            # outputs["identity_selection_g/{}".format(scale)] = idxs_g.float()  # 只读

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_pg += to_optimise_pg.mean()

            losses["loss_pg/{}".format(scale)] = loss_pg

            if self.opt.disparity_smoothness != 0 and scale == 0:  # smooth就只放一个scale 0 的就行了
                smooth_loss = get_smooth_loss(norm_disp, color)  # b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s = 0

            total_loss += loss_pg + loss_s

        total_loss /= self.num_scales
        losses["loss"] = total_loss  # mean of 4 scales
        return losses

    def compute_losses_test_bp(self, inputs, outputs):
        """
        单独测试identity loss， 其他为0, 确实不可学习， 参数应该是动都没动
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_pg = 0
            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0, scale)]
            color = inputs[("color", 0, scale)]  # ??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]



            # auto masking Enable/Disable
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))  # total [b,2,h,w], 2 means pred frame and post frame
                # 直接用相邻帧做损失, loss很小的地方说明后三种静止
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)  # list2batch

            identity_reprojection_loss = identity_reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            combined_p = identity_reprojection_loss


            to_optimise_pg,idxs = torch.min(combined_p,dim=1)  ##b4hw, weights, 越大loss越小



            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_pg += to_optimise_pg.mean()

            losses["loss_pg/{}".format(scale)] = loss_pg

            if self.opt.disparity_smoothness != 0 and scale == 0:  # smooth就只放一个scale 0 的就行了
                smooth_loss = get_smooth_loss(norm_disp, color)  # b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s = 0

            total_loss += loss_pg + loss_s

        total_loss /= self.num_scales
        losses["loss"] = total_loss  # mean of 4 scales
        return losses

    def compute_losses_test_bp2(self, inputs, outputs):
        """
        只考虑repro loss， identity loss 置0
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss_pg = 0
            reprojection_losses = []
            geometry_losses = []

            source_scale = 0

            disp = outputs[("disp", 0, scale)]
            color = inputs[("color", 0, scale)]  # ??
            target = inputs[("color", 0, source_scale)]
            pred_depth = outputs[("depth", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:  # 前一帧,后一帧,两次计算
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)  # list b,1,h,w  to b2hw



            # avg reprojection loss Enable/Disable
            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            combined_p = reprojection_loss

            # to_optimise_g, idxs_g = torch.min(combined_g,dim=1)#idxs_g应该和


            # if self.opt.softmin:
            to_optimise_pg ,idxs_pg= torch.min(combined_p,dim=1)




            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # geometry loss add
            loss_pg += to_optimise_pg.mean()

            losses["loss_pg/{}".format(scale)] = loss_pg

            if self.opt.disparity_smoothness != 0 and scale == 0:  # smooth就只放一个scale 0 的就行了
                smooth_loss = get_smooth_loss(norm_disp, color)  # b1hw,b3hw
                loss_s = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                losses["loss_s/{}".format(scale)] = loss_s
            else:
                loss_s = 0

            total_loss += loss_pg + loss_s

        total_loss /= self.num_scales
        losses["loss"] = total_loss  # mean of 4 scales
        return losses

    def compute_depth_metrics(self, inputs, outputs):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance

        in1 outputs[("depth", 0, 0)]
        in2 inputs["depth_gt"]
        out1 losses
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [self.opt.full_height, self.opt.full_width], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop#????
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)
        metrics={}
        for i, metric in enumerate(self.depth_metric_names):
            metrics[metric] = np.array(depth_errors[i].cpu())
        return  metrics

    def terminal_log(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def tb_log(self, mode, inputs=None, outputs=None, losses=None,metrics=None):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        if losses!=None:
            for l, v in losses.items():
                writer.add_scalar("{}".format(l), v, self.step)
        if metrics!=None:
            for l,v in metrics.items():
                writer.add_scalar("{}".format(l), v, self.step)
        if inputs!=None and outputs!=None:
            for j in range(min(4, self.opt.batch_size)):  # write a maxminmum of four images
                for s in self.opt.scales:
                    for frame_id in self.opt.frame_ids:
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step)
                        if s == 0 and frame_id != 0:
                            writer.add_image(
                                "color_pred_{}_{}/{}".format(frame_id, s, j),
                                outputs[("color", frame_id, s)][j].data, self.step)

                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp", 0,s)][j]), self.step)

                    if "identity_selection/{}".format(s) in outputs.keys():
                        img = tensor2array(outputs["identity_selection/{}".format(s)][j],colormap='bone')
                        writer.add_image(
                                "automask_{}/{}".format(s, j),
                                #outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
                                img, self.step)#add 1,h,w

                    if "identity_selection_g/{}".format(s) in outputs.keys():
                        writer.add_image(
                            "automask_g{}/{}".format(s,j),
                            outputs["identity_selection_g/{}".format(s)][j][None, ...], self.step)


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = self.checkpoints_path/"models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(self.checkpoints_path, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """

        save_folder = self.checkpoints_path/"models"/"weights_{}".format(self.epoch)
        save_folder.makedirs_p()

        for model_name, model in self.models.items():
            save_path = save_folder/ "{}.pth".format(model_name)
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = save_folder/ "{}.pth".format("adam")
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
    #main cycle
    def epoch_train(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            #model forwardpass
            outputs, losses = self.process_batch(inputs)#

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.tb_log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            #
            self.logger.train_logger_update(batch= batch_idx,time = duration,names=losses.keys(),values=[item.cpu().data for item in losses.values()])

            #val, and terminal_val_log, and tb_log
            if early_phase or late_phase:
                #self.terminal_log(batch_idx, duration, losses["loss"].cpu().data)


                #metrics={}
                #if "depth_gt" in inputs:
                #    metrics = self.compute_depth_metrics(inputs, outputs)
                self.tb_log(mode="train", inputs=inputs, outputs=outputs, losses =losses)#terminal log
                if "depth_gt" in inputs:
                    self.metrics = self.compute_depth_metrics(inputs, outputs)
                    self.tb_log(mode='train', metrics=self.metrics)
                self.val()


            self.step += 1


        self.logger.reset_train_bar()
        self.logger.reset_valid_bar()

            #record the metric

    #only 2 methods for public call
    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            epc_st = time.time()
            self.epoch_train()
            duration = time.time() - epc_st
            self.logger.epoch_logger_update(epoch=self.epoch,time=duration,names=self.metrics.keys(),values=["{:.4f}".format(float(item)) for item in self.metrics.values()])
            if (self.epoch + 1) % self.opt.weights_save_frequency == 0 :
                self.save_model()

    @torch.no_grad()
    def val(self):
        """Validate the model on a single minibatch
        这和之前的常用框架不同， 之前是在train all batches 后再 val all batches，
        这里train batch 再 val batch（frequency）
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()
        time_st = time.time()
        outputs, losses = self.process_batch(inputs)
        duration =time.time() -  time_st
        self.logger.valid_logger_update(batch=self.val_iter.rcvd_idx,time=duration,names=losses.keys(),values=[item.cpu().data for item in losses.values()])



        self.tb_log(mode="val", inputs=inputs, outputs=outputs, losses=losses)

        if "depth_gt" in inputs:
            metrics = self.compute_depth_metrics(inputs, outputs)
            self.tb_log(mode="val",metrics = metrics)
            del metrics

        del inputs, outputs, losses
        self.set_train()
