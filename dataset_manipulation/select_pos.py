import numpy as np
import os
import cv2
import natsort

if __name__ == "__main__":
    #####################
    #       MAIN        #
    #####################

    mask_path = '/home/bh/Downloads/data_2111/blood/'
    raw_path = '/home/bh/Downloads/data_2111/raw/'

    mask_dst_path = '/home/bh/Downloads/data_2111/blood_pos/'
    raw_dst_path = '/home/bh/Downloads/data_2111/raw_blood_pos/'

    file_list = natsort.natsorted(os.listdir(mask_path))

    subject_idx = np.arange(1,51)

    for sub_idx in subject_idx:
        print(sub_idx)

        for file_idx in file_list:
            if file_idx == '.DS_Store': continue
            idx_split = file_idx.split('_')
            if int(idx_split[0])==int(sub_idx):
                mask = cv2.imread(os.path.join(mask_path, file_idx), cv2.IMREAD_GRAYSCALE)
                raw = cv2.imread(os.path.join(raw_path, file_idx), cv2.IMREAD_GRAYSCALE)

                mask_uniq = np.unique(mask)
                if len(mask_uniq) == 2:
                    # print(mask_uniq)
                    cv2.imwrite(os.path.join(mask_dst_path, file_idx), mask)
                    cv2.imwrite(os.path.join(raw_dst_path, file_idx), raw)

