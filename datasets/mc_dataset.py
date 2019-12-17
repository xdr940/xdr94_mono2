

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from path import Path
import matplotlib.pyplot as plt
from .mono_dataset import MonoDataset


class MCDataset(MonoDataset):
    def __init__(self,*args,**kwargs):
        super(MCDataset,self).__init__(*args,**kwargs)

        #self.full_res_shape = [1920,1080]#
        self.full_res_shape = [800,600]#


        #FOV = 35d

        #960/fx = tan 35 =0.7-> fx = 1371

        # 1920 * k[0] = 1371-> k0 = 0.714
        # 1080 * k[1 ]= 1371 -> k1 = 1.27
        self.K=np.array([[0.714, 0, 0.5, 0],
                           [0, 1.27, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)


        self.img_ext='.png'
        self.depth_ext = '.png'

        self.MaxDis = 255
        self.MinDis = 0


    def check_depth(self):

        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        depth_filename =Path(self.data_path)/scene_name/"depth"/"{:07d}.png".format(int(frame_index))

        return depth_filename.exists()

    def get_color(self, folder, frame_index, side, do_flip):
        path =self.get_image_path(folder, frame_index)
        color = self.loader(path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


    def get_image_path(self, folder, frame_index):
        f_str = "{:07d}{}".format(frame_index, self.img_ext)
        image_path = Path(self.data_path)/ folder/"img/{}".format(f_str)
        return image_path



    def get_depth(self, folder, frame_index, side, do_flip):
        path = self.get_depth_path(folder, frame_index)
        depth_gt = plt.imread(path)
        depth_gt = skimage.transform.resize(depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt#[0~1]

    def get_depth_path(self, folder, frame_index):
        f_str = "{:07d}{}".format(frame_index, self.img_ext)
        depth_path = Path(self.data_path) / folder / "depth/{}".format(f_str)
        return depth_path
