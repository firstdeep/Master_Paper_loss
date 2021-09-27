import os
import natsort
import ast
import numpy as np
import cv2


if __name__ =="__main__":

    # mask_dir = '/home/bh/digits/for_detectnet/test'
    # for i in range(1,5):
    #
    #     f = open(os.path.join(mask_dir, 'fold_%d.txt'%(i)), 'w')
    #     file_list = natsort.natsorted(os.listdir(os.path.join(mask_dir, 'fold%d_test'%(i))))
    #
    #     for idx in file_list:
    #         file_path = os.path.join(mask_dir, 'fold%d_test'%(i), idx)
    #         f.write(str(file_path)+'\n')
    #     f.close()


    file_path = '/home/bh/digits/for_detectnet/test/fold_4_result.txt'
    dst_path = '/home/bh/digits/for_detectnet/test/'

    img_path = os.path.join(dst_path, 'fold4_test')
    img_list = natsort.natsorted(os.listdir(img_path))

    f = open(file_path, 'r')

    total_list = []

    for i in f:
        box_list = []

        idx_split = i.split('\t')
        box = idx_split[4]
        box = box[2:].split('.')

        box_list.append(int(box[0]))
        box_list.append(int(box[1]))
        box_list.append(int(box[2]))
        box_list.append(int(box[3]))
        total_list.append(box_list)


    f.close()
    print(len(total_list))
    total_list = np.array(total_list)
    np.save(os.path.join(dst_path, 'fold4_result_0909.npy'), total_list)

    # # length * 4(top[x,y], bottom[x,y])
    # fold2_test_npy = np.load(os.path.join(dst_path, 'fold2_test.npy'))


    for idx, img_idx in enumerate(img_list):
        img = cv2.imread(os.path.join(img_path, img_idx), cv2.IMREAD_COLOR)
        bbox = total_list[idx]
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0),3)

        cv2.imwrite(os.path.join(dst_path, 'fold4_rec', img_idx), img)