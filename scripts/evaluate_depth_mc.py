from __future__ import absolute_import, division, print_function

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from networks.layers import disp_to_depth
from utils.official import readlines
from opts.md_eval_opts import MD_eval_opts
import datasets
import networks
from tqdm import  tqdm
from path import Path
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)



# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
SCALE_FACTOR = 800.


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

@torch.no_grad()
def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 800.
    #这里的度量信息是强行将gt里的值都压缩到和scanner一样的量程， 这样会让值尽量接近度量值
    #但是对于

    if not opt.eval_mono or opt.eval_stereo:print("Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo")

    splits_dir = Path(opt.root)/'splits'

    #1. load gt
    print('\n-> load gt\n')
    gt_path = Path(splits_dir) / opt.eval_split / "gt_depths.npz"
    gt_depths = np.load(gt_path,allow_pickle=True)
    gt_depths = gt_depths["data"]


    #2. load img data and predict, output is pred_disps(shape is [nums,1,w,h])

    depth_eval_path = Path(opt.depth_eval_path)

    if not depth_eval_path.exists():print("Cannot find a folder at {}".format(depth_eval_path))


    print("-> Loading weights from {}".format(depth_eval_path))


    #model loading
    filenames = readlines(splits_dir/opt.eval_split/ "test_files.txt")
    encoder_path = depth_eval_path/"encoder.pth"
    decoder_path = depth_eval_path/ "depth.pth"

    encoder_dict = torch.load(encoder_path)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    # dataloader
    dataset = datasets.MCDataset(opt.data_path,
                                       filenames,
                                       encoder_dict['height'],
                                       encoder_dict['width'],
                                       [0], 4, is_train=False)
    dataloader = DataLoader(dataset,
                            batch_size=opt.eval_batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            pin_memory=True,
                            drop_last=False)


    pred_disps = []

    print("\n-> Computing predictions with size {}x{}\n".format(
        encoder_dict['width'], encoder_dict['height']))

    #prediction
    for data in tqdm(dataloader):
        input_color = data[("color", 0, 0)].cuda()

        if opt.post_process:
            # Post-processed results require each image to have two forward passes
            input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

        output = depth_decoder(encoder(input_color))

        pred_disp, pred_depth = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
        pred_disp = pred_disp.cpu()[:, 0].numpy()
        #pred_depth = pred_depth.cpu()[:,0].numpy()
        if opt.post_process:
            N = pred_disp.shape[0] // 2
            pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

        pred_disps.append(pred_disp)
    #endfor
    pred_disps = np.concatenate(pred_disps)



    #


    #3. evaluation
    #gt_depths [nums,600,800]
    #pred_disps [nums,288,384]
    print("\n-> Evaluating")


    opt.pred_depth_scale_factor = SCALE_FACTOR
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    nums_evaluate = pred_disps.shape[0]

    for i in tqdm(range(nums_evaluate)):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))#1271x341 t0 192x640
        pred_depth = 1 / pred_disp# 也可以根据上面直接得到

        #crop
        mask = gt_depth >0

        pred_depth = pred_depth[mask]#并reshape成1d
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor

        #median scaling
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)#中位数， 在eval的时候， 将pred值线性变化，尽量能使与gt接近即可
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH#所有历史数据中最小的depth, 更新,
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH#...

        errors.append(compute_errors(gt_depth, pred_depth))

    '''
    *255
   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.284  &  14.421  &  40.817  &   0.402  &   0.622  &   0.802  &   0.881  \\
    
    *1
      abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.284  &   0.057  &   0.160  &   0.402  &   0.622  &   0.802  &   0.881  \\

sq_rel and rmse度量的时候需要进行量级统一
    
*800.
abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.208  &   17.591  &  57.653  &   0.402  &   0.622  &   0.802  &   0.881  \\

    
    '''

    #4. precess results, latex style output
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print("\n Scaling ratios | med: {:0.3f} | std: {:0.3f}\n".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")




if __name__ == "__main__":
    options = MD_eval_opts()
    evaluate(options.parse())
