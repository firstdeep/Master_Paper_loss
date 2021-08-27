import numpy as np
import os
import cv2
import natsort

if __name__ == "__main__":
    #####################
    #       MAIN        #
    #####################

    mask_path = '/home/bh/Desktop/0826_Data/512/mask_all/'
    file_list = natsort.natsorted(os.listdir(mask_path))

    subject_idx = np.arange(1,63)

    sub_1 = np.arange(52,74)
    sub_48_fron = np.arange(3,15)
    sub_48_back = np.arange(28,34)

    for sub_idx in subject_idx:

        idx_list = []
        for file_idx in file_list:
            idx_split = file_idx.split('_')
            if int(idx_split[0])==int(sub_idx):
                mask = cv2.imread(os.path.join(mask_path, file_idx), cv2.IMREAD_GRAYSCALE)

                mask[mask>0] = 255
                cv2.imwrite(os.path.join(mask_path, file_idx), mask)

                mask_uniq = np.unique(mask)

                if len(mask_uniq) == 2:
                    if sub_idx == 1:
                        if int(idx_split[1].split('.')[0]) in sub_1:
                            mask = mask * 0
                            cv2.imwrite(os.path.join(mask_path, file_idx), mask)

                    if sub_idx == 48:
                        if int(idx_split[1].split('.')[0]) in sub_48_fron or int(idx_split[1].split('.')[0]) in sub_48_back:
                            mask = mask * 0
                            cv2.imwrite(os.path.join(mask_path, file_idx), mask)

                    idx_list.append(int(idx_split[1].split('.')[0]))

        if len(idx_list) != (idx_list[-1] - idx_list[0] + 1):
            print("*"*100)
            print(sub_idx)
        else:
            print("length of list = %d, sub = %d"%(len(idx_list), (idx_list[-1] - idx_list[0] + 1)))

