import aug as am
import glob
import random
import os
from shp2imagexy import shp2imagexy


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def generate_img(image_dir,
                 crop_dir,
                 out_dir,
                 small_num=2):
    mkdir(out_dir)
    imglist = glob.glob(f'{image_dir}/*/*.tif')
    small_imglist = glob.glob(f'{crop_dir}/*.tif')
    print(f'the length of img is {len(imglist)}')
    print(f'the length of small img is {len(small_imglist)}')
    random.shuffle(imglist)
    random.shuffle(small_imglist)
    for imgPath in imglist:
        subRoot = os.path.split(imgPath)[0]
        shpPath = glob.glob(f'{subRoot}/*.shp')[0]
        small_img = random.sample(small_imglist, small_num)
        labels, boxes = shp2imagexy(imgPath, shpPath)
        am.copysmallobjects2(imgPath, labels, boxes, out_dir, small_img, class_dict={'1':1})


if __name__ == '__main__':
    from shutil import copyfile, rmtree
    image_dir = r'sheepfold'
    crop_dir = r'crop_images'
    out_dir = 'enhance_images'
    generate_img(image_dir, crop_dir, out_dir)
    filesList = os.listdir(image_dir)
    for file in filesList:
        file = os.path.join(image_dir, file)
        new_file = file.replace('V1', 'V2')
        if os.path.exists(new_file):
            rmtree(new_file)
        mkdir(new_file)
        subFilesList = os.listdir(file)
        for subfile in subFilesList:
            suffix = subfile.split('.')[-1]
            baseName = subfile.split('.')[0].split('_')[0]
            subNewFile = os.path.join(new_file, f'{baseName}_v2.{suffix}')
            subfile = os.path.join(file, subfile)
            if suffix == 'tif':
                subfile = f'{out_dir}/{baseName}.tif'
                copyfile(subfile, subNewFile)
            else:
                copyfile(subfile, subNewFile)
