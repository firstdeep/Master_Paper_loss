import numpy as np
import cv2

check_dir = '../AAAGilDatasetPos/'
subject = '05390853_20200821'
img_idx = 83

while(1):
    raw_name = check_dir + '/raw/' + subject + '_%04d.png'%img_idx
    mask_name = check_dir + '/mask/' + subject + '_%04d.png'%img_idx
    raw = cv2.imread(raw_name, 1)
    mask = cv2.imread(mask_name, 0)
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    mask_rgb[:,:,1] = mask
    cv2.imshow('raw', raw + (mask_rgb/2).astype(np.uint8))
    cv2.imshow('mask_rgb', mask_rgb)
    if cv2.waitKey(0) == 'q':
        break
    img_idx = img_idx + 1