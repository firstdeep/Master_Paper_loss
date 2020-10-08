import os
import numpy as np
from shutil import copyfile
import cv2

src_dir = '/home/hyoseok/research/medical/aaa/dataset/'
tgt_pos_dir = '/home/hyoseok/research/medical/aaa/aaa_mask/AAAGilDatasetPos/'
tgt_neg_dir = '/home/hyoseok/research/medical/aaa/aaa_mask/AAAGilDatasetNeg/'

major = 0
minor = 0

dir_list = os.listdir(src_dir + '/raw_zinv/')

for entry in dir_list:
    minor = 0
    file_list = os.listdir(src_dir + '/raw_zinv/' + entry)
    file_list.sort()
    for f in file_list:
        if 'png' in f:
            src_img = src_dir + '/raw_zinv/' + entry + '/' + f
            src_mask = src_dir + '/segment_post/' + entry + '/' + f


            mask = cv2.imread(src_mask, 0)
            if np.sum(mask) > 255:
                tgt_img = tgt_pos_dir + '/raw/' + f
                tgt_mask = tgt_pos_dir + '/mask/' + f

            else:
                tgt_img = tgt_neg_dir + '/raw/' + f
                tgt_mask = tgt_neg_dir + '/mask/' + f

            copyfile(src_img, tgt_img)
            copyfile(src_mask, tgt_mask)





