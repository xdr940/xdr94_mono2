#move files
from path import Path
import re
from utils.official import readlines
import os
from tqdm import  tqdm

def MC():
    cp_img=False
    cp_gt =True
    dataset = Path("/home/roit/datasets/MC")

    wk_root = Path('/home/roit/aws/aprojects/xdr94_mono2')
    root = wk_root / 'splits/mc/test_files.txt'

    img_dump = wk_root/'mc_test_img'
    img_dump.mkdir_p()

    gt_dump = wk_root/'mc_test_gt'
    gt_dump.mkdir_p()


    files = readlines(root)

    for item in tqdm(files):
        block,p,color,frame = item.split('/')
        if cp_img:
            img_p = dataset/block/p/'color'/frame+'.png'
            out_name = item.replace('/','_')+'.png'
            cmd = 'cp '+img_p+'  '+img_dump/out_name
            os.system(cmd)
        if cp_gt:
            gt_p =  dataset/block/p/'depth'/frame+'.png'
            out_name = item.replace('/', '_') + '.png'
            cmd = 'cp ' + gt_p + '  ' + gt_dump / out_name
            os.system(cmd)




def kitti():
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
    MC()