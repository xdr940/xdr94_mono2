

from datasets.custom_mono import CustomMono
from torch.utils.data import DataLoader
from utils.official import *
import matplotlib.pyplot as plt
from path import Path
import argparse


parser = argparse.ArgumentParser(
    description='Simple testing funtion for Monodepthv2 models.')

parser.add_argument('--dataset_path', type=str, default='/home/roit/datasets/Binjiang',
                    help='path to a test image or folder of images')
parser.add_argument("--num_workers", default=1, )
parser.add_argument('--out_path', type=str, default=None, help='path to a test image or folder of images')
parser.add_argument("--scales", default=[0,1,2,3])
parser.add_argument("--frame_idxs", default=[-1,0,1], help="train, val, test")
parser.add_argument("--batch_size", default=12)
parser.add_argument("--height", default=192)
parser.add_argument("--width", default=640)


args = parser.parse_args()


def main(args):
    dataset_path = Path(args.dataset_path)
    splits = 'custom_mono'
    fpath = Path(os.path.dirname(__file__))/ "splits"/splits/ "{}_files.txt"
    train_filenames = readlines(fpath.format("train"))
    val_filenames = readlines(fpath.format("val"))
    img_ext = '.jpg'

    train_dataset = CustomMono(
        data_path=dataset_path,
        filenames=train_filenames,
        height=args.height,
        width=args.width,
        frame_idxs=args.frame_idxs,
        num_scales=len(args.scales)
    )

    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )


    for batch_idx,inputs in enumerate(dataloader):
        print('ok')


if __name__=="__main__":
    main(args)