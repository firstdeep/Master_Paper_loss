import os
import numpy as np
from shutil import copyfile, move
import cv2
import natsort

mask_dir = '/home/bh/Desktop/AAA_DATA_NEW/512/mask_all'
raw_dir = '/home/bh/Desktop/AAA_DATA_NEW/512/raw_all'

mask_dst_dir = '/home/bh/Desktop/AAA_DATA_NEW/512/mask_all'
raw_dst_dir = '/home/bh/Desktop/AAA_DATA_NEW/512/raw_all'

list_file = natsort.natsorted(os.listdir(mask_dir))

for file_idx in list_file:

    idx_split = file_idx.split('_')

    # if int(idx_split[0]) == 1:
    img_mask = cv2.imread(os.path.join(mask_dir, file_idx), cv2.IMREAD_GRAYSCALE)

    total_sum = img_mask.sum()
    if total_sum !=0 and total_sum/255 < 100:
        # print(file_idx)
        img_mask = img_mask * 0
        print(np.unique(img_mask))
        cv2.imwrite(os.path.join(mask_dir, file_idx), img_mask)

