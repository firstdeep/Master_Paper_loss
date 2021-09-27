import numpy as np
import os
import cv2
import natsort
import math


# Draw elipsis on image
def fit_ellipse(mask):
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    # find biggest contour
    # contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    # biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    biggest_contour = max(contours, key=cv2.contourArea)

    cnt = biggest_contour
    ellipse = cv2.fitEllipse(cnt)

    # Calculate ellipse diameter
    (xc, yc), (d1, d2), angle = ellipse
    rmajor = max(d1, d2) / 2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90
    xtop = xc + math.cos(math.radians(angle)) * rmajor
    ytop = yc + math.sin(math.radians(angle)) * rmajor
    xbot = xc + math.cos(math.radians(angle + 180)) * rmajor
    ybot = yc + math.sin(math.radians(angle + 180)) * rmajor

    x = int(xtop) - int(xbot)
    y = int(ytop) - int(ybot)
    ellipse_diameter = math.sqrt(math.pow(x,2) + math.pow(y,2))
    bigger_pixel_ellip = ellipse_diameter

    x,y,w,h = cv2.boundingRect(cnt)

    bigger_pixel_rect = w
    if bigger_pixel_rect < h:
         bigger_pixel_rect = h


    return bigger_pixel_rect, int(bigger_pixel_ellip)



if __name__ == "__main__":
#     #####################
#     #       MAIN        #
#     #####################
#
#     mask_path = '/home/bh/Downloads/aaa_segmentation/0803/result_0819_update_all/'
#
#     # GT
#     # mask_path = '/home/bh/Desktop/AAA_DATA_NEW/256/mask_all'
#
#     file_list = natsort.natsorted(os.listdir(mask_path))
#     file_list.pop()
#
#     data_arr = np.zeros((51,300))
#
#     subject_idx = np.arange(1,52)
#
#     count = 0
#
#     for sub_idx in subject_idx:
#         ellipse = []
#         print(sub_idx)
#         for file_idx in file_list:
#
#             idx_split = file_idx.split('_')
#             num = int(idx_split[1].split('.')[0])
#
#             if int(idx_split[0]) == sub_idx:
#                 mask = cv2.imread(os.path.join(mask_path, file_idx), cv2.IMREAD_GRAYSCALE)
#                 # Check positive sample
#                 pos_check = int(len(np.unique(mask)))
#
#                 if pos_check == 2:
#                     print(file_idx)
#                     if file_idx == "1_0070.png" or file_idx == "2_0067.png":
#                         continue
#                     rect, ellip_diameter = fit_ellipse(mask)
#                     ### ellipse check
#                     ellipse.append(ellip_diameter)
#
#                     if ellip_diameter >=18:
#                         data_arr[(sub_idx-1),count] = 1
#
#                 count = count + 1
#         # print(data_arr[(sub_idx-1)])
#         # print(ellipse)
#         count = 0
#
#
# np.save("./predict_default_18",data_arr)
# print("*"*50)
# print("==== Done ====")

#####################
#       MAIN        #
#####################

    # mask_path = '/home/bh/Downloads/aaa_segmentation/0803/result_0819_update_all/'

    npy_path = "/home/bh/digits/for_detectnet/test"

    fold1 = np.load(os.path.join(npy_path,'fold1_result_0909.npy'))
    fold2 = np.load(os.path.join(npy_path,'fold2_result_0909.npy'))
    fold3 = np.load(os.path.join(npy_path,'fold3_result_0909.npy'))
    fold4 = np.load(os.path.join(npy_path,'fold4_result_0909.npy'))

    fold1_list = natsort.natsorted(os.listdir(os.path.join(npy_path, "fold1_test")))
    fold2_list = natsort.natsorted(os.listdir(os.path.join(npy_path, "fold2_test")))
    fold3_list = natsort.natsorted(os.listdir(os.path.join(npy_path, "fold3_test")))
    fold4_list = natsort.natsorted(os.listdir(os.path.join(npy_path, "fold4_test")))

    data_arr = np.zeros((60, 400))

    subject_idx1 = np.arange(1, 16)
    subject_idx2 = np.arange(16, 31)
    subject_idx3 = np.arange(31, 46)
    subject_idx4 = np.arange(46, 61)

    count = 0

    for sub_idx in subject_idx1:
        # print(sub_idx)
        for idx, file_idx in enumerate(fold1_list):

            idx_split = file_idx.split('_')
            num = int(idx_split[1].split('.')[0])

            if int(idx_split[0]) == sub_idx:

                bbox = fold1[idx]

                if sum(bbox) != 0:
                    print(file_idx)
                    data_arr[(sub_idx - 1), count] = 1

                count = count + 1

        count = 0

    for sub_idx in subject_idx2:
        # print(sub_idx)
        for idx, file_idx in enumerate(fold2_list):

            idx_split = file_idx.split('_')
            num = int(idx_split[1].split('.')[0])

            if int(idx_split[0]) == sub_idx:

                bbox = fold2[idx]

                if sum(bbox) != 0:
                    print(file_idx)
                    data_arr[(sub_idx - 1), count] = 1

                count = count + 1

        count = 0

    for sub_idx in subject_idx3:
        # print(sub_idx)
        for idx, file_idx in enumerate(fold3_list):

            idx_split = file_idx.split('_')
            num = int(idx_split[1].split('.')[0])

            if int(idx_split[0]) == sub_idx:

                bbox = fold3[idx]

                if sum(bbox) != 0:
                    print(file_idx)
                    data_arr[(sub_idx - 1), count] = 1

                count = count + 1

        count = 0

    for sub_idx in subject_idx4:
        # print(sub_idx)
        for idx, file_idx in enumerate(fold4_list):

            idx_split = file_idx.split('_')
            num = int(idx_split[1].split('.')[0])

            if int(idx_split[0]) == sub_idx:

                bbox = fold4[idx]

                if sum(bbox) != 0:
                    print(file_idx)
                    data_arr[(sub_idx - 1), count] = 1

                count = count + 1

        count = 0

    np.save("./predict_detectnet_512", data_arr)
    print("*" * 50)
    print("==== Done ====")