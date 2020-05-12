```python

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



            # auto masking 04250958 完全不学习     
         
```python
         
          map_0, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
            rhosvar = erro_maps.var(dim=1,unbiased=False)#bhw4 --> BHW
            rhosvar_flat = rhosvar.flatten(start_dim=1)#B,H*W

            #rhosvar normalization
            #max norm
            #max,_ = rhosvar_flat.max(dim=1)#B
            #rhosvar_flat = rhosvar_flat/max.unsqueeze(1)#b,hw / b,1 = b,h*w

            #mean norm
            median,_ = rhosvar_flat.median(dim=1)#b
            rhosvar_flat /= median.unsqueeze(1)#b,1 / b,1

            delta_var,_ = rhosvar_flat.median(dim=1)#B
            var_mask = (rhosvar_flat > delta_var.unsqueeze(1)).reshape_as(rhosvar)

            final_selection = rhosvar.float()* var_mask.float()
            to_optimise = map_0 *final_selection#loss map
            #to_optimise = map_0 * var_mask.float()#loss map
            #to_optimise = reprojection_loss.float()#loss map



            outputs["to_optimise/{}".format(scale)] = to_optimise.float()

            outputs["rhosvar/{}".format(scale)] = rhosvar.float()


            outputs["var_mask/{}".format(scale)] = var_mask.float()
            outputs["final_selection/{}".format(scale)] = final_selection




            #---------------------



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
``` Enable/Disable
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
```


```python
            04221133
#-------------------------
            delta =0
            map_0, idxs_0 = torch.min(erro_maps, dim=1)  # b,rho,h,w
            #erro_maps_norm = torch.softmax(-erro_maps, dim=1)  # normalization
            var = erro_maps.var(dim=1)

            map_2 = map_0 * var
            var_mask = (var > delta)
            to_optimise = map_2 * var_mask.float()
            mo_ob = (var < delta)

            identity_selection = (idxs_0 > identity_reprojection_loss.shape[1] - 1)
            final_selection = identity_selection * var_mask

            #outputs["identity_selection/{}".format(scale)] = identity_selection.float()

            outputs["mo_ob/{}".format(scale)] = mo_ob.float()

            outputs["final_selection/{}".format(scale)] = final_selection.float()

            #---------------------
```

04221411

```python

#-------------------------
            delta =0.0001
            map_0, idxs_0 = torch.min(erro_maps, dim=1)  # b,rho,h,w
            erro_maps_norm = torch.softmax(-erro_maps, dim=1)  # normalization
            var = erro_maps_norm.var(dim=1)

            var_mask = (var > delta)
            to_optimise = map_0 * var_mask.float()
            #mo_ob = (var < delta)

            identity_selection = (idxs_0 > identity_reprojection_loss.shape[1] - 1)
            final_selection = identity_selection * var_mask

            outputs["identity_selection/{}".format(scale)] = idxs_0.float()

            outputs["mo_ob/{}".format(scale)] = var_mask.float()

            outputs["final_selection/{}".format(scale)] = final_selection.float()

            #---------------------
```


04230050

```python

           map_0, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
            rhosvar = erro_maps.var(dim=1)#BHW
            rhosvar = rhosvar.flatten(start_dim=1)#B,H*W
            max,_ = rhosvar.max(dim=1)#B
            mean = rhosvar.mean(dim=1)#B
            rhosvar = rhosvar/max.unsqueeze(1)#b,hw / b,1
            var_mask = (rhosvar > mean.unsqueeze(1)).reshape_as(map_0)
            to_optimise = map_0 * var_mask.float()
            #mo_ob = (var < delta)

            identity_selection = (idxs_0 > identity_reprojection_loss.shape[1] - 1)
            final_selection = identity_selection * var_mask

            outputs["identity_selection/{}".format(scale)] = idxs_0.float()

            outputs["mo_ob/{}".format(scale)] = var_mask.float()

            outputs["final_selection/{}".format(scale)] = final_selection.float()

            #---------------------
            
            
```


```python

#-------------------------
            map_0, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
            rhosvar = erro_maps.var(dim=1,unbiased=False)#BHW
            rhosvar = rhosvar.flatten(start_dim=1)#B,H*W
            
            max,_ = rhosvar.max(dim=1)#B
            mean = rhosvar.mean(dim=1)#B
            rhosvar = rhosvar/max.unsqueeze(1)#b,hw / b,1
            var_mask = (rhosvar > mean.unsqueeze(1)).reshape_as(map_0)
            to_optimise = map_0 * var_mask.float()
            #mo_ob = (var < delta)

            identity_selection = (idxs_0 > identity_reprojection_loss.shape[1] - 1)
            final_selection = identity_selection * var_mask

            outputs["identity_selection/{}".format(scale)] = identity_selection.float()

            outputs["var_mask/{}".format(scale)] = var_mask.float()

            outputs["final_selection/{}".format(scale)] = final_selection.float()

            #---------------------
            
```


202004240123
```python

#-------------------------
            map_0, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw

            rhosvar = erro_maps.var(dim=1,unbiased=False)#BHW
            rhosvar = rhosvar.flatten(start_dim=1)#B,H*W
            delta_var = rhosvar.mean(dim=1)#B
            #rhosvar normalization
            max,_ = rhosvar.max(dim=1)#B
            rhosvar = rhosvar/max.unsqueeze(1)#b,hw / b,1

            #rhosmean
            rhosmean = erro_maps.mean(dim=1)#BHW
            rhosmean = rhosmean.flatten(start_dim=1)
            delta_mean = rhosmean.mean(dim=1)


            var_mask = (rhosvar > delta_var.unsqueeze(1)).reshape_as(map_0)
            #var_mask = (rhosvar > .1).reshape_as(map_0)

            mean_mask = (rhosmean > delta_mean.unsqueeze(1)).reshape_as(map_0)
            #mean mask : 1 说明为moving region

            ind_mov = (1-var_mask)*  mean_mask

            static = (1-var_mask)* (1-mean_mask)

            to_optimise = map_0 * var_mask.float()
            #mo_ob = (var < delta)

            identity_selection = (idxs_0 > identity_reprojection_loss.shape[1] - 1)
            final_selection = identity_selection * var_mask
            outputs["ind_mov/{}".format(scale)] = ind_mov.float()

            outputs["static/{}".format(scale)] = static.float()

            outputs["to_optimise/{}".format(scale)] = to_optimise.float()

            outputs["rhosvar/{}".format(scale)] = rhosvar.reshape_as(map_0).float()

            outputs["identity_selection/{}".format(scale)] = identity_selection.float()

            outputs["mean_mask/{}".format(scale)] = mean_mask.float()
            outputs["var_mask/{}".format(scale)] = var_mask.float()


            outputs["final_selection/{}".format(scale)] = final_selection.float()

            #---------------------
            

```
还是不能用soft， 值太小没必要.
```python


#-------------------------
            erro_maps = erro_maps.transpose(2,1).transpose(2,3)#bhw4
            map_0, idxs_0 = torch.min(erro_maps, dim=3)  # b,h,w,4-->bhw,bhw

            rhosvar = erro_maps.var(dim=3,unbiased=False)#bhw4 --> BHW
            rhosvar_flat = rhosvar.flatten(start_dim=1)#B,H*W
            delta_var = rhosvar_flat.mean(dim=1)#B

            #rhosvar normalization
            max,_ = rhosvar_flat.max(dim=1)#B
            rhosvar_flat = rhosvar_flat/max.unsqueeze(1)#b,hw / b,1
            #



            var_mask = (rhosvar_flat > delta_var.unsqueeze(1)).reshape_as(rhosvar)

            to_optimise = map_0 *rhosvar.unsqueeze(3)* var_mask.unsqueeze(3).float()



        #    outputs["to_optimise/{}".format(scale)] = to_optimise.float()

            outputs["rhosvar/{}".format(scale)] = rhosvar.float()


            outputs["var_mask/{}".format(scale)] = var_mask.float()



            #---------------------
```  
    04250958 完全不学习     
         
```python
         
          map_0, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
            rhosvar = erro_maps.var(dim=1,unbiased=False)#bhw4 --> BHW
            rhosvar_flat = rhosvar.flatten(start_dim=1)#B,H*W

            #rhosvar normalization
            #max norm
            #max,_ = rhosvar_flat.max(dim=1)#B
            #rhosvar_flat = rhosvar_flat/max.unsqueeze(1)#b,hw / b,1 = b,h*w

            #mean norm
            median,_ = rhosvar_flat.median(dim=1)#b
            rhosvar_flat /= median.unsqueeze(1)#b,1 / b,1

            delta_var,_ = rhosvar_flat.median(dim=1)#B
            var_mask = (rhosvar_flat > delta_var.unsqueeze(1)).reshape_as(rhosvar)

            final_selection = rhosvar.float()* var_mask.float()
            to_optimise = map_0 *final_selection#loss map
            #to_optimise = map_0 * var_mask.float()#loss map
            #to_optimise = reprojection_loss.float()#loss map



            outputs["to_optimise/{}".format(scale)] = to_optimise.float()

            outputs["rhosvar/{}".format(scale)] = rhosvar.float()


            outputs["var_mask/{}".format(scale)] = var_mask.float()
            outputs["final_selection/{}".format(scale)] = final_selection




            #---------------------



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
```
04251246

```python


 map_0, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
            rhosvar = erro_maps.var(dim=1,unbiased=False)#bhw4 --> BHW
            rhosvar_flat = rhosvar.flatten(start_dim=1)#B,H*W

            #rhosvar normalization
            #max norm
            #max,_ = rhosvar_flat.max(dim=1)#B
            #rhosvar_flat = rhosvar_flat/max.unsqueeze(1)#b,hw / b,1 = b,h*w

            #mean norm
            median,_ = rhosvar_flat.median(dim=1)#b
            rhosvar_flat /= median.unsqueeze(1)#b,1 / b,1

            delta_var,_ = rhosvar_flat.median(dim=1)#B
            var_mask = (rhosvar_flat > delta_var.unsqueeze(1)).reshape_as(rhosvar)

            #final_selection =  var_mask.float()#rhosvar.float()
            final_selection = rhosvar*(1- var_mask.float()) + var_mask.float()
            to_optimise = map_0 *final_selection#loss map
            #to_optimise = map_0 * var_mask.float()#loss map
            #to_optimise = reprojection_loss.float()#loss map



            outputs["to_optimise/{}".format(scale)] = to_optimise.float()

            outputs["rhosvar/{}".format(scale)] = 1- var_mask.float()


            outputs["var_mask/{}".format(scale)] = var_mask.float()
            #outputs["final_selection/{}".format(scale)] = final_selection




            #---------------------
```


```python
04261212
#-------------------------
            map_0, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
            rhosvar = erro_maps.var(dim=1,unbiased=False)#bhw4 --> BHW
            rhosvar_flat = rhosvar.flatten(start_dim=1)#B,H*W

            #rhosvar normalization
            #max norm
            #max,_ = rhosvar_flat.max(dim=1)#B
            #rhosvar_flat = rhosvar_flat/max.unsqueeze(1)#b,hw / b,1 = b,h*w

            #mean norm
            median,_ = rhosvar_flat.median(dim=1)#b
            rhosvar_flat /= median.unsqueeze(1)#b,1 / b,1

            delta_var,_ = rhosvar_flat.median(dim=1)#B
            delta_var = delta_var.unsqueeze(1)
            delta_var = 0.5
            var_mask = (rhosvar_flat > delta_var).reshape_as(rhosvar)

            #final_selection =  var_mask.float()#rhosvar.float()
            #final_selection = rhosvar*(1- var_mask.float()) + var_mask.float()
            to_optimise = map_0 *var_mask.float()#loss map
            #to_optimise = map_0 * var_mask.float()#loss map
            #to_optimise = reprojection_loss.float()#loss map
            identity_selection = (idxs_0 > identity_reprojection_loss.shape[1] - 1)



            outputs["to_optimise/{}".format(scale)] = to_optimise.float()

            #outputs["rhosvar/{}".format(scale)] = 1- var_mask.float()
            outputs["identity_selection/{}".format(scale)] = identity_selection.float()

            outputs["var_mask/{}".format(scale)] = var_mask.float()
            #outputs["final_selection/{}".format(scale)] = final_selection




            #---------------------
```

试试identical mask 遮掩， 理论上应该和min完全一样
```python

 def compute_losses_f(self, inputs, outputs):
        """通过var 计算移动物体
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", 0, scale)]
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

            erro_maps = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)#b4hw

            # -------------------------
            map_0, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw

            rhosvar = erro_maps.var(dim=1, unbiased=False)  # BHW
            rhosvar_flat = rhosvar.flatten(start_dim=1)  # B,H*W
            median, _ = rhosvar_flat.median(dim=1)  # b
            #rhosvar_flat /= median.unsqueeze(1)
            delta_var = rhosvar_flat.mean(dim=1).unsqueeze(1)  # B

            var_mask = (rhosvar_flat > 0.001).reshape_as(map_0)
            same_mask = (rhosvar_flat<delta_var/10).reshape_as(map_0)

            # rhosmean
            rhosmean = erro_maps.mean(dim=1)  # BHW
            rhosmean_flat = rhosmean.flatten(start_dim=1)#b,h*w
            delta_mean = rhosmean_flat.mean(dim=1).unsqueeze(dim=1)#b,1
            mean_mask = (rhosmean_flat <2* delta_mean).reshape_as(map_0)

            # mean mask : 1 说明为moving region
            #ind_mov = (1 - var_mask) * mean_mask

            #static = (1 - var_mask) * (1 - mean_mask)
            identity_selection = (idxs_0 >= 2)#

            final_mask = var_mask.float()*mean_mask.float()*identity_selection.float()
            to_optimise = map_0 * final_mask
            # mo_ob = (var < delta)

            #identity_selection = (idxs_0 > identity_reprojection_loss.shape[1] - 1)

            outputs["identity_selection/{}".format(scale)] = identity_selection.float()
            outputs["mean_mask/{}".format(scale)] = mean_mask.float()
            outputs["var_mask/{}".format(scale)] = var_mask.float()



            outputs["final_selection/{}".format(scale)] = final_mask.float()

            # ---------------------

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


```
identical mask

```python

 # -------------------------
            map_0, idxs_0 = torch.min(erro_maps, dim=1)  # b,4,h,w-->bhw,bhw
            map1, idxs_1 = torch.min(reprojection_loss,dim=1)

            #rhosvar = erro_maps.var(dim=1, unbiased=False)  # BHW
            #rhosvar_flat = rhosvar.flatten(start_dim=1)  # B,H*W
            #median, _ = rhosvar_flat.median(dim=1)  # b
            #rhosvar_flat /= median.unsqueeze(1)
            #delta_var = rhosvar_flat.mean(dim=1).unsqueeze(1)  # B

            #var_mask = (rhosvar_flat > 0.001).reshape_as(map_0)
            #same_mask = (rhosvar_flat<delta_var/10).reshape_as(map_0)

            # rhosmean
            #rhosmean = erro_maps.mean(dim=1)  # BHW
            #rhosmean_flat = rhosmean.flatten(start_dim=1)#b,h*w
            #delta_mean = rhosmean_flat.mean(dim=1).unsqueeze(dim=1)#b,1
            #mean_mask = (rhosmean_flat <2* delta_mean).reshape_as(map_0)

            # mean mask : 1 说明为moving region
            #ind_mov = (1 - var_mask) * mean_mask

            #static = (1 - var_mask) * (1 - mean_mask)
            identity_selection = (idxs_0 >= 2)#

            #final_mask = var_mask.float()*mean_mask.float()*identity_selection.float()
            to_optimise = map1 * identity_selection.float()
            # mo_ob = (var < delta)

            #identity_selection = (idxs_0 > identity_reprojection_loss.shape[1] - 1)

            outputs["identity_selection/{}".format(scale)] = identity_selection.float()
            #outputs["mean_mask/{}".format(scale)] = mean_mask.float()
            #outputs["var_mask/{}".format(scale)] = var_mask.float()



            #outputs["final_selection/{}".format(scale)] = final_mask.float()

            # ---------------------
```