# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse
from path import Path
file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default='/home/roit/datasets/kitti/')
                                 #default="/home/roit/datasets/MC")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default='/media/roit/hard_disk_2/Models/monodepth2/checkpoints')

        self.parser.add_argument('--root',type=str,default='/home/roit/aws/aprojects/xdr94_mono2')

        # TRAINING options

        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "custom",'custom_small',"eigen_full", "odom", "benchmark","mc","mc_small"],
                                 default="custom_small")
                                 #default="mc_small")

        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 #default="mc",
                                 default='kitti',
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "mc"])


        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 default=True,
                                 action="store_true")
        self.parser.add_argument("--height",type=int,help="input image height",default=192)
        self.parser.add_argument("--width",type=int,help="input image width",default=640)
        self.parser.add_argument("--full_height",type=int,
                                 default=375)
                                 #default = 600)

        self.parser.add_argument("--full_width",type=int,
                                 default=1242)
                                 #default = 800)

        self.parser.add_argument("--disparity_smoothness",type=float,help="disparity smoothness weight",default=0.1)
        self.parser.add_argument("--histc_weights",type=float,help="disparity smoothness weight",default=0)
        self.parser.add_argument("--geometry_loss_weights",default=0.,type=float)

        self.parser.add_argument("--scales",nargs="+",type=int,help="scales used in the loss",default=[0, 1, 2, 3])

        self.parser.add_argument("--min_depth",type=float,help="minimum depth",default=0.1)#这里度量就代表m
        self.parser.add_argument("--max_depth",type=float,help="maximum depth",default=80.0)

        #self.parser.add_argument("--use_stereo",help="if set, uses stereo pair for training",action="store_true")
        self.parser.add_argument("--frame_ids",nargs="+",type=int,help="frames to load",default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",type=int,help="batch size",default=8)#
        self.parser.add_argument("--learning_rate",type=float,help="learning rate",default=1e-4)
        self.parser.add_argument("--num_epochs",type=int,help="number of epochs",default=20)
        self.parser.add_argument("--scheduler_step_size",type=int,help="step size of the scheduler",default=15)

        # ABLATION options
        self.parser.add_argument("--softmin",default=False)

        #self.parser.add_argument("--automasking",
        #                         default=True,
        #                         help="if set, doesn't do auto-masking",
        #                         action="store_true")
        #self.parser.add_argument("--ssim",
        #                         help="if set, enables ssim in the loss",
        #                         default=True,
        #                         action="store_true")

        #self.parser.add_argument("--v1_multiscale",
        #                         help="if set, uses monodepth v1 multiscale",
        #                         default=False,
        #                         action="store_true")

        #self.parser.add_argument("--avg_reprojection",
        #                         help="if set, uses average reprojection loss",
        #                         default=False,
        #                         action="store_true")
        #self.parser.add_argument("--predictive_mask",
        #                         help="if set, uses a predictive masking scheme as in Zhou et al",
        #                         default=False,
        #                         action="store_true")



        #other set?
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="/home/roit/models/torchvision/resnet18.pth",
                                 choices=["pretrained", "scratch"])
        #self.parser.add_argument("--pose_model_input",
        #                         type=str,
        #                         help="how many images the pose network gets",
        #                         default="pairs",
        #                         choices=["pairs", "all"])
        #self.parser.add_argument("--pose_model_type",
        #                         type=str,
        ##                         help="normal or shared",
         #                        default="separate_resnet",#supplementary 中已经说了separate_resnet精度更好,ablation 在 table12
         #                        choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 default='/home/roit/models/monodepth2/mono_640x192',
                                 #default=None,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--tb_log_frequency",
                                 type=int,
                                 help="number of batches(step) between each tensorboard log",
                                 default=5)
        self.parser.add_argument("--weights_save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--depth_eval_path",default='/media/roit/hard_disk_2/Models/monodepth2/checkpoints/12-17-23:15/models/weights_19')

        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 default=True,
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        #self.parser.add_argument("--ext_disp_to_eval",
        #                         type=str,
        #                         help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="mc",#eigen
                                 choices=["eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10","custom","mc"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",default='eval_out_dir',
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",#??
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

        self.parser.add_argument("--eval_pose_data_path",
                                 default='/media/roit/hard_disk_2/Datasets/kitti_odometry_color')

        self.parser.add_argument("--eval_pose_save_path",default="./")
        self.parser.add_argument("--eval_batch_size",default=16,type=int)
        self.parser.add_argument("--eval_odom_batch_size",default=16,type=int)





    def parse(self):
        self.options = self.parser.parse_args()
        return self.options


class MCOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # TEST MCDataset

        self.parser.add_argument("--data_path",
                                 default="/home/roit/datasets/MC")
        self.parser.add_argument("--height", default=192)
        self.parser.add_argument("--width", default=256)
        self.parser.add_argument("--frame_idxs",default=[-1,0,1])
        self.parser.add_argument("--scales",default=[0,1,2,3])

        self.parser.add_argument("--batch_size",default=1)
        self.parser.add_argument("--num_workers",default=1)

        self.parser.add_argument("--splits",default='MC')



    def parse(self):
        self.options = self.parser.parse_args()
        return self.options