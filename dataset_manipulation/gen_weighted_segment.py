import numpy as np
import cv2
import os



src_dir = '/home/hyoseok/research/medical/aaa/aaa_mask/AAAGilDatasetPos/mask/'
tgt_dir = '/home/hyoseok/research/medical/aaa/aaa_mask/AAAGilDatasetPos/mask_weighted/'

if not os.path.isdir(tgt_dir):
    os.mkdir(tgt_dir)

file_list = os.listdir(src_dir)

for f in file_list:
    if 'png' in f:
        src_file = src_dir + f
        tgt_file = tgt_dir + f
        print(tgt_file)

        img_src = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
        img_dist = cv2.distanceTransform(img_src, cv2.DIST_L2, 3)
        img_src_float = img_src.astype(np.float32)
        img_src_float = img_src_float - img_dist
        img_weighted_seg = np.ceil(img_src_float).astype(np.uint8)

        img_weighted_seg[(img_weighted_seg > 0) & (img_weighted_seg<250)] = 250

        cv2.imwrite(tgt_file, img_weighted_seg)
