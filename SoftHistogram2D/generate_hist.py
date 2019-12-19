import  torch

from path import Path
import numpy as np

import matplotlib.pyplot as plt
from SoftHistogram2D.soft_hist import SoftHistogram2D_H,SoftHistogram2D_W
import argparse
from tqdm import  tqdm
import cv2
parser =argparse.ArgumentParser(description="generate histogram ")
parser.add_argument('--path',default='/home/roit/datasets/depth_out/visdrone_test_out')
parser.add_argument("--dataset",choices=['mc','kitti','visdrone'],default='visdrone')
parser.add_argument("--data_from",choices=['model_out','gt'],default='model_out')
parser.add_argument('--out_path',default=None)
parser.add_argument('--resize',default=False)
parser.add_argument('--out_h',default=256,type=int)
parser.add_argument('--out_w',default=384,type=int)
parser.add_argument('--ext',default=None)
parser.add_argument("--out_ext",default='png')

parser.add_argument('--type',default='rgb',choices=['rgb','gray'])
parser.add_argument('--scales',default=255,choices=[1,255])

args = parser.parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def generate_histogram():

    # dir mak
    in_path  = Path(args.path)
    if args.out_path ==None:
        outp=Path( './'+in_path.stem+"_histc")
        outp.mkdir_p()
        outh_path =outp/'outh'
        outv_path =outp/'outv'
    else:
        outp = Path(args.out_path)
        outp.mkdir()
        outh_path = outp/'outh'
        outv_path = outp/'outv'

    outh_path.mkdir_p()
    outv_path.mkdir_p()

    #get shape
    if args.ext!=None:
        files = in_path.files('*.{}'.format(args.ext))
    else:
        files = in_path.files()
    files.sort()
    img = plt.imread(files[0])

    if len(img.shape)==3:#rgb2gray
        togray = [0.299, 0.587, 0.114]
        img =np.dot( img[:,:,:3],togray)


    h,w = img.shape
    if args.dataset =='kitti' or 'visdrone':#因为尺寸不一， 故resize成一致的
        args.resize=True
        args.out_w = 640
        args.out_h=192

    #funtion mk
    c = 1
    b = 1

    if args.resize:
        hfunc = SoftHistogram2D_H(device=device,bins=255,min=0,max=255,sigma=3,h=args.out_h,w=args.out_w)
        vfunc = SoftHistogram2D_W(device=device,bins=255,min=0,max=255,sigma=3,h=args.out_h,w=args.out_w)

    else:
        hfunc = SoftHistogram2D_H(device=device,bins=255,min=0,max=255,sigma=3,h=h,w=w)
        vfunc= SoftHistogram2D_W(device=device,bins=255,min=0,max=255,sigma=3,h=h,w=w)

    #
    i=0
    for img_p in tqdm(files):
        img = plt.imread(img_p)

        if img.max() <= 1:  # [0~1]->[1,255]
            img *= 255

        if len(img.shape)==3:
            togray = [0.299, 0.587, 0.114]
            img = np.dot(img[:, :, :3], togray)
        if args.data_from != 'gt':
            img  = 255 - img

        #if args.scales==255:
        #    img/=255

        if args.resize:
            h,w = img.shape
            fx = args.out_w / w
            fy = args.out_h / h
            img = cv2.resize(img,(0,0),fx=fx,fy=fy,interpolation=cv2.INTER_CUBIC)

        img=torch.tensor(img,dtype=torch.float).to(device)
        outh = hfunc(img)
        outv=vfunc(img)
        plt.imsave(outh_path/img_p.stem+'.{}'.format(args.out_ext),outh.detach().cpu().numpy())
        plt.imsave(outv_path/img_p.stem+'.{}'.format(args.out_ext),outv.detach().cpu().numpy())



if __name__ == '__main__':
    generate_histogram()
