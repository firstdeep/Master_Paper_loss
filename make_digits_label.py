import os
import natsort
import cv2
import numpy as np

if __name__=="__main__":

    work_dir = '/home/bh/Downloads/0906_modify_full_contrast/0906_rename_for_bh'
    mask_dir = os.path.join(work_dir, 'mask_all')

    file_idx = natsort.natsorted(os.listdir(mask_dir))

    if not os.path.exists(os.path.join(work_dir, 'label')):
        os.makedirs(os.path.join(work_dir, 'label'))

    for idx in file_idx:

        idx_split = idx.split('.')
        mask = cv2.imread(os.path.join(mask_dir, idx), cv2.IMREAD_GRAYSCALE)


        # check B.B coordinate
        if len(np.unique(mask)) == 2:
            pos = np.where(mask)
            l = np.min(pos[1]) - 5
            t = np.min(pos[0]) - 5
            r = np.max(pos[1]) + 5
            b = np.max(pos[0]) + 5

        else:
            l = 20
            t = 20
            r = 30
            b = 30

        # img = cv2.imread(os.path.join(work_dir, 'raw_all', idx), cv2.IMREAD_COLOR)
        # cv2.rectangle(img, (l,t), (r,b), (0,255,0), 3)
        # cv2.imwrite(os.path.join(work_dir,'raw_bbox_check', idx), img)
        # cv2.imshow("img", img)
        # cv2.waitKey(1000000)

        with open(work_dir + '/label/' + idx_split[0] + '.txt', 'w') as fp:
                if len(np.unique(mask)) == 2:
                    type = 'Car'
                    truncated = 0
                    occluded = 3
                    alpha = 0
                    tail = '0 0 0 0 0 0 0 0'

                    label = type + ' ' + \
                            str(truncated) + ' ' + \
                            str(occluded) + ' ' + \
                            str(alpha) + ' ' + \
                            str(l) + ' ' + str(t) + ' ' + str(r) + ' ' + str(b) + ' ' + tail

                    fp.write(label + '\n')

                else:
                    type = 'DontCare'
                    truncated = -1
                    occluded = -1
                    alpha = -10
                    tail = '-1 -1 -1 -1000 -1000 -1000 -10'

                    label = type + ' ' + \
                            str(truncated) + ' ' + \
                            str(occluded) + ' ' + \
                            str(alpha) + ' ' + \
                            str(l) + ' ' + str(t) + ' ' + str(r) + ' ' + str(b) + ' ' + tail

                    fp.write(label + '\n')
