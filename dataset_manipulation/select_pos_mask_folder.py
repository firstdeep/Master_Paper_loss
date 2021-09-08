import numpy as np
import os
import cv2
import natsort
from shutil import copyfile

if __name__ == "__main__":
    #####################
    #       MAIN        #
    #####################

    mask_path = '/home/bh/Downloads/0906_modify_window_contrast/ROI/'
    raw_path = '/home/bh/Downloads/0906_modify_window_contrast/RAW_PNG/'

    mask_dst_path = '/home/bh/Downloads/0906_modify_window_contrast/ROI_pos/'
    raw_dst_path = '/home/bh/Downloads/0906_modify_window_contrast/RAW_PNG_pos/'

    file_list = natsort.natsorted(os.listdir(mask_path))


    for sub_idx in file_list:
        if sub_idx=='.DS_Store': continue

        if not os.path.exists(os.path.join(mask_dst_path, sub_idx)):
            os.makedirs(os.path.join(mask_dst_path, sub_idx))
            os.makedirs(os.path.join(raw_dst_path, sub_idx))

        sub_file_list = natsort.natsorted(os.listdir(os.path.join(mask_path, sub_idx)))
        for file_idx in sub_file_list:
            if file_idx=='.DS_Store': continue

            idx_split = file_idx.split('_')

            mask = cv2.imread(os.path.join(mask_path, sub_idx,file_idx), cv2.IMREAD_GRAYSCALE)
            mask_uniq = np.unique(mask)

            if len(mask_uniq) == 2:
                copyfile(os.path.join(mask_path, sub_idx, file_idx), os.path.join(mask_dst_path, sub_idx, file_idx))
                raw_img_name = idx_split[0]+'_'+idx_split[1]+'_'+idx_split[3]
                copyfile(os.path.join(raw_path, sub_idx, raw_img_name), os.path.join(raw_dst_path, sub_idx, raw_img_name))


