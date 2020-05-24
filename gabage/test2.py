#move files
from path import Path
import re
from utils.official import readlines
import os
from tqdm import  tqdm



def main():
    dataset = Path("/home/roit/datasets/kitti")

    wk_root = Path('/home/roit/aws/aprojects/xdr94_mono2')
    root = wk_root / 'splits/eigen/test_files.txt'

    out_path = wk_root / 'eigen_test_img'
    out_path.mkdir_p()
    files = readlines(root)
    for item in tqdm(files):
        dir,pre,num,lr = re.split(' |/',item)
        if lr =='l':
            out_name = pre +'_'+ num+'_'+lr+'.png'
            cmd = 'cp '+ dataset/dir/pre/'image_02/data'/num+'.png'+ ' '+out_path/out_name
            os.system(cmd)

    print('ok')

if __name__ == '__main__':
    main()