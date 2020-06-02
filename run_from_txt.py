# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
from path import Path
import glob
import numpy as np
import PIL.Image as pil
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.official import readlines
import torch
from torchvision import transforms, datasets
import path
import networks
from layers import disp_to_depth
from utils.official import download_model_if_doesnt_exist

from options import run_infer_from_txt
#parse_args_run_from_txt  as parse_args
@torch.no_grad()
def main(args):
    """Function to predict for a single image or folder of images
    """
    print(args.image_path)
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")



    #download_model_if_doesnt_exist(args.model_path,args.model_name)

    model_path = Path(args.model_path)/ args.model_name
    if not model_path.exists():
        print(model_path+" does not exists")

    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

#1 LOADING PRETRAINED MODEL
    #1.1 encoder
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    #1.2 decoder
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

#2. FINDING INPUT IMAGES

    in_path = Path(args.image_path)

    #files
    root = Path(os.path.dirname(__file__))
    txt = root/'splits'/args.txt_style/'test_files.txt'
    in_files = readlines(txt)
    #out
    if args.out_path !=None:
        out_path  =Path(args.out_path)
    else:
        out_path = Path('./'+in_path.stem+'_out')
    out_path.mkdir_p()
    files=[]
    if args.txt_style =='custom' or args.txt_style =='eigen':#kitti
        for item in  in_files:
            item = item.split(' ')
            if item[2]=='l':camera ='image_02'
            elif item[2]=='r': camera= 'image_01'
            files.append(in_path/item[0]/camera/'data'/"{:010d}.png".format(int(item[1])))
    elif args.txt_style =='mc':
        for item in  in_files:
            #item = item.split('/')
            files.append(in_path/(item +'.png'))
    elif args.txt_style =='visdrone':
        for item in in_files:
            item = item.split(' ')
            files.append(in_path / 'sequences' / item[0] / item[1]+'.jpg')
#2.1

    cnt=0
#3. PREDICTING ON EACH IMAGE IN TURN
    print('\n-> inference '+args.image_path)
    for  image_path in tqdm(files):



        # Load image and preprocess
        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)#torch.Size([1, 3, 192, 640])
        features = encoder(input_image)#a list from 0 to 4
        outputs = depth_decoder(features)# dict , 4 disptensor
        cnt+=1
        disp = outputs[("disp", 0)]# has a same size with input
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving numpy file
        #if args.out_name=='num':
        if args.txt_style=='eigen' or args.txt_style=='custom':
            output_name = str(image_path).split('/')[-4]+'_{}'.format(image_path.stem)
        else:
            output_name = str(image_path).split('/')[-3]+'_{}'.format(image_path.stem)


        if args.npy_out:
            name_dest_npy = os.path.join(out_path, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        name_dest_im =Path(out_path)/"{}.png".format(output_name)
        plt.imsave(name_dest_im, disp_resized_np, cmap='magma', vmax=vmax)

    print(cnt)

    print('\n-> Done,save at '+args.out_path)


if __name__ == '__main__':
    options = run_infer_from_txt()
    main(options.parse())
