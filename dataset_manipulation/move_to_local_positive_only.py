import os
import numpy as np
from shutil import copyfile
import cv2

src_dir = '/home/hyoseok/research/medical/aaa/dataset/'
tgt_dir = '/home/hyoseok/research/medical/aaa/aaa_mask/AAAGilDatasetPos/'

major = 0
minor = 0

dir_list = os.listdir(src_dir + '/raw')

for entry in dir_list:
    minor = 0
    file_list = os.listdir(src_dir + '/raw/' + entry)
    file_list.sort()
    for f in file_list:
        if 'png' in f:
            src_img = src_dir + '/raw/' + entry + '/' + f
            src_mask = src_dir + '/segment_post/' + entry + '/' + f


            mask = cv2.imread(src_mask, 0)
            if np.sum(mask) > 255:

                img_name = 'img_%04d_%04d.png' % (major, minor)
                mask_name = 'mask_%04d_%04d.png' % (major, minor)

                tgt_img = tgt_dir + '/img/' + img_name
                tgt_mask = tgt_dir + '/mask/' + mask_name

                copyfile(src_img, tgt_img)
                copyfile(src_mask, tgt_mask)

                minor = minor + 1

                print(tgt_img)
                print(tgt_mask)


    major = major + 1
