import argparse


def parse_args_run_from_txt():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        #default='/home/roit/datasets/MC',
                        default='/home/roit/datasets/kitti',
                        help='path to a test image or folder of images')
    parser.add_argument("--txt_style",
                        #default='mc',
                        default='eigen',
                        choices=['custom','mc','visdrone','eigen','mc_small'])
    parser.add_argument('--out_path', type=str,default='eigen_test_out',help='path to a test image or folder of images')
    parser.add_argument('--npy_out',default=False)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        default='mono_640x192',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--model_path',type=str,default='/home/roit/models/monodepth2',help='root path of models')
    parser.add_argument('--ext', type=str,help='image extension to search for in folder', default="*.jpg")
    parser.add_argument("--no_cuda",help='if set, disables CUDA',action='store_true')
    parser.add_argument("--out_ext",default="*.png")

    return parser.parse_args()