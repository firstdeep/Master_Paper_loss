import os
import numpy as np
from shutil import copyfile, move
import cv2
import natsort

mask_dir = '/home/bh/Desktop/AAA_DATA_NEW/512/mask'
raw_dir = '/home/bh/Desktop/AAA_DATA_NEW/512/raw'

mask_dst_dir = '/home/bh/Desktop/AAA_DATA_NEW/512/mask_all'
raw_dst_dir = '/home/bh/Desktop/AAA_DATA_NEW/512/raw_all'

mask_pos_dir = '/home/bh/Desktop/AAA_DATA_NEW/512/mask_pos'
raw_pos_dir = '/home/bh/Desktop/AAA_DATA_NEW/512/raw_pos'

list_file = natsort.natsorted(os.listdir(mask_dst_dir))
# list_folder.pop()

subject = np.arange(1,52)
sub = 1
count = 0

for file_idx in list_file:
    split_file = file_idx.split('_')

    mask = cv2.imread(os.path.join(mask_dst_dir, file_idx), cv2.IMREAD_GRAYSCALE)
    raw = cv2.imread(os.path.join(raw_dst_dir, file_idx), cv2.IMREAD_GRAYSCALE)

    if sub == int(split_file[0]):

        if int(len(np.unique(mask))) == 2:
            print(file_idx)
            idx = "{0:04d}".format(count)
            rename = str(sub)+"_"+str(idx)+".png"
            cv2.imwrite(os.path.join(mask_pos_dir, rename),mask)
            cv2.imwrite(os.path.join(raw_pos_dir, rename), raw)
            count = count + 1

    elif sub != int(split_file[0]):
        sub = sub+1
        count = 0

        if int(len(np.unique(mask))) == 2:
            idx = "{0:04d}".format(count)
            rename = str(sub)+"_"+str(idx)+".png"
            cv2.imwrite(os.path.join(mask_pos_dir, rename),mask)
            cv2.imwrite(os.path.join(raw_pos_dir, rename), raw)
            count = count + 1






