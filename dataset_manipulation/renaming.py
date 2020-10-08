import os
import numpy as np
from shutil import copyfile, move
import cv2

src_dir = '/home/hyoseok/research/medical/aaa/dataset/'
tgt_dir = '/home/hyoseok/research/medical/aaa/dataset/'

major = 0
minor = 0

dir_list = os.listdir(src_dir + '/raw')

for entry in dir_list:
    minor = 0
    file_list = os.listdir(src_dir + '/raw/' + entry)
    for f in file_list:
        if 'png' in f:
            src_img1 = src_dir + '/raw/' + entry + '/' + f

            names = f.split('.')
            nums = names[0].split('_')
            nums[2] = '%04d'%np.int32(nums[2])

            f2 = '%s_%s_%s.%s'%(nums[0], nums[1], nums[2], names[1])

            src_img1 = src_dir + '/raw/' + entry + '/' + f
            src_img2 = src_dir + '/raw/' + entry + '/' + f2

            src_mask1 = src_dir + '/segment_post/' + entry + '/' + f
            src_mask2 = src_dir + '/segment_post/' + entry + '/' + f2

            move(src_img1, src_img2)
            move(src_mask1, src_mask2)

            # src_img = src_dir + '/raw/' + entry + '/' + f
            # src_mask = src_dir + '/segment_post/' + entry + '/' + f