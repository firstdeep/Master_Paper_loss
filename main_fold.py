import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from GilAAADataset import *
from engine import train_one_epoch, evaluate
import utils
import pickle
import cv2
import transforms as T
from torchvision.transforms import functional as F
from sklearn.model_selection import KFold
from torchvision.models.detection import roi_heads
from torchsummary import summary as summary_
import natsort

import pandas as pd

from gil_eval import *
import random
from unet import mask_unet

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    model.roi_heads.mask_unet = mask_unet()
    return model


def main(mode, model_path_name, gpu_idx=0, train_batch_size=1):


    GPU_NUM = gpu_idx
    fold_num = 4
    device = torch.device(f'cuda:{GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')

    # raw_path = 'data/update/all'
    raw_path = 'data/update/pos'
    # raw_path = 'data/1_NEW/256_all'

    total_dataset = GilAAADataset(raw_path, get_transform(train=True))
    total_dataset_test = GilAAADataset(raw_path, get_transform(train=False))

    ################################
    # Modify subject range
    total_subject = list(range(1,52))
    kfold = KFold(n_splits=fold_num, shuffle=False)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(total_subject)):
        if fold!=0:
            continue

        for index, value in enumerate(test_ids):
            test_ids[index] = value + 1
        test_idx = []

        index = 0
        for path in total_dataset.imgs:
            split_path = path.split("_")
            if int(split_path[0]) in test_ids:
                test_idx.append(index)

            index = index + 1

        total_index = list(range(0, len(total_dataset.imgs)))
        train_idx = [index for index in total_index if index not in test_idx]

        indices1 = train_idx
        indices2 = test_idx

        np.random.shuffle(indices1)
        np.random.shuffle(indices2)

        valid_idx = random.sample(indices1, int(len(indices1)*0.1))
        indices1 = [index for index in indices1 if index not in valid_idx]

        dataset = torch.utils.data.Subset(total_dataset, indices1)
        dataset_valid = torch.utils.data.Subset(total_dataset_test, valid_idx)
        dataset_test = torch.utils.data.Subset(total_dataset_test, indices2)

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=train_batch_size, shuffle=True, num_workers=0,
            collate_fn=utils.collate_fn)

        data_loader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=utils.collate_fn)

        # our dataset has two classes only - background and ...
        num_classes = 2

        # get the model using our helper function
        model = get_instance_segmentation_model(num_classes)
        # move model to the right device
        model.to(device)


        if 'train' in mode:
            print("*" * 30)
            print("\n")
            print("\t KFOLD:%d Training ..."%fold)
            print("\n")
            print("*" * 30)

            # construct an optimizer
            params = [p for p in model.parameters() if p.requires_grad]

            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
            # optimizer = torch.optim.Adam(params, lr=0.001)
            # optimizer = torch.optim.RMSprop(params, lr=0.0002)

            # and a learning rate scheduler which decreases the learning rate by
            # 10x every 3 epochs
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

            # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, threshold=0.2)

            # let's train it for 10 epochs
            num_epochs = 10

            for epoch in range(num_epochs):

                # train for one epoch, printing every 10 iterations
                train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

                # validation
                val_loss = evaluate(model, data_loader_valid, device=device)
                print(val_loss)

                # update the learning rate
                # lr_scheduler.step(val_loss)
                lr_scheduler.step()

                if((epoch+1)%10 == 0):
                    torch.save(model.state_dict(), './pretrained/256_s_fold_%d_%s.pth' % (fold, model_path_name))
                    torch.save(dataset_test, './pretrained/256_s_test_fold_%d_%s.pth' % (fold, model_path_name))

    if 'test' in mode:
        print("*"*25)
        print("\n")
        print("\t\tTesting ...")
        print("\n")
        print("*" * 25)

        if not os.path.exists("/home/bh/Downloads/aaa_segmentation/0803/result_%s/"%(model_path_name)):
            os.mkdir("/home/bh/Downloads/aaa_segmentation/0803/result_%s/"%(model_path_name))
        if not os.path.exists('/home/bh/Downloads/aaa_segmentation/0803/result_%s/result_analysis/'%(model_path_name)):
            os.mkdir("/home/bh/Downloads/aaa_segmentation/0803/result_%s/result_analysis/"%(model_path_name))
        save_dir = '/home/bh/Downloads/aaa_segmentation/0803/result_%s/'%(model_path_name)
        save_dir_analysis = '/home/bh/Downloads/aaa_segmentation/0803/result_%s/result_analysis/'%(model_path_name)

        total_ol = []
        total_ja = []
        total_di = []
        total_fp = []
        total_fn = []

        for fold, (train_ids, test_ids) in enumerate(kfold.split(total_subject)):
            # if fold!= 0:
            #     continue

            for index, value in enumerate(test_ids):
                test_ids[index] = value + 1

            fold_ol = []
            fold_ja = []
            fold_di = []
            fold_fp = []
            fold_fn = []

            model.load_state_dict(torch.load('./pretrained/256_s_fold_%d_%s.pth'%(fold,model_path_name)))
            # test_path = "0815_update_all"
            dataset_test = torch.load('./pretrained/256_s_test_fold_%d_%s.pth'%(fold,model_path_name))

            num_test = len(dataset_test.indices)

            for i in range(num_test):
                img_name = raw_path + '/raw/' + dataset_test.dataset.imgs[dataset_test.indices[i]]
                mask_name = raw_path + '/mask/' + dataset_test.dataset.imgs[dataset_test.indices[i]]

                img = Image.open(img_name).convert("RGB")
                mask_gt = Image.open(mask_name).convert("RGB")
                img_rgb = np.array(img)
                img = F.to_tensor(img)

                model.eval()
                with torch.no_grad():
                    prediction = model([img.to(device)])

                if (list(prediction[0]['boxes'].shape)[0] == 0):
                    mask = np.zeros((256, 256), dtype=np.uint8)
                else:
                    mask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())

                img_mask = np.array(mask)
                img_mask_gt = np.array(mask_gt)
                img_mask[img_mask > 127] = 255
                img_mask[img_mask <= 127] = 0

                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

                img_gray_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

                img_overlap = img_mask_gt.copy()
                img_overlap[:, :, 0] = 0
                img_overlap[:, :, 1] = img_mask

                img_pred_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)

                add_img = cv2.addWeighted(img_gray_color, 0.7, img_overlap, 0.3, 0)

                red_gt = img_mask_gt.copy()
                green_pred = img_pred_color.copy()

                idx_gt = np.where(red_gt > 0)
                idx_sr = np.where(green_pred > 0)
                red_gt[idx_gt[0], idx_gt[1], :] = [0, 0, 255]
                green_pred[idx_sr[0], idx_sr[1], :] = [0, 255, 0]

                cv2.putText(img_gray_color, "\"Raw image\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)
                cv2.putText(add_img, "\"Raw + GT + Predict\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)
                cv2.putText(red_gt, "\"GT\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)
                cv2.putText(green_pred, "\"Predict\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)
                cv2.putText(img_overlap, "\"GT + predict\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)

                img_all = np.concatenate([img_gray_color, add_img, red_gt, green_pred, img_overlap], axis=1)

                cv2.imwrite(save_dir_analysis + dataset_test.dataset.imgs[dataset_test.indices[i]].replace('.png', '_maskrcnn.png'), img_all)
                cv2.imwrite(save_dir + dataset_test.dataset.imgs[dataset_test.indices[i]], img_mask)

            for subject in test_ids:
                # print("Fold = %d, subject = %d"%(fold, subject))
                overlap, jaccard, dice, fn, fp = eval_segmentation_volume(save_dir, str(subject), raw_path)
                print(str(subject) + ' %.4f %.4f %.4f %.4f %.4f' % (overlap, jaccard, dice, fn, fp))

                # print("=" * 50)
                # print("\n")
                fold_ol.append(overlap)
                fold_ja.append(jaccard)
                fold_di.append(dice)
                fold_fn.append(fn)
                fold_fp.append(fp)
            total_ol.append(np.mean(fold_ol))
            total_ja.append(np.mean(fold_ja))
            total_di.append(np.mean(fold_di))
            total_fn.append(np.mean(fold_fn))
            total_fp.append(np.mean(fold_fp))

        print('[Average volume evaluation] overlap:%.4f jaccard:%.4f dice:%.4f fn:%.4f fp:%.4f' % (
            np.mean(total_ol), np.mean(total_ja), np.mean(total_di), np.mean(total_fn), np.mean(total_fp)))


    if 'detection' in mode:
        print("*"*25)
        print("\n")
        print("\t\tDetection ...")
        print("\n")
        print("*" * 25)
        shape = (53,1)
        total_list = np.zeros(shape)
        total_list = list(total_list)

        gt_path = os.path.join(raw_path, "mask")
        pred_path = '/home/bh/Downloads/aaa_segmentation/0803/result_%s/'%(model_path_name)

        list_file = natsort.natsorted(os.listdir(gt_path))

        subject = 1
        gt_list = []
        pred_list = []
        file_list = []
        total_rate = []
        pred_file_idx = []

        # fold_1 = list([1,2,3,4,5,6,7,8,9,10,11,12,13,14])

        for idx in list_file:
            idx_spilt = idx.split('_')

            # if int(idx_spilt[0]) not in fold_1:
            #     continue

            if subject != int(idx_spilt[0]):
                print("%d" % (int(idx_spilt[0])-1))
                # print(len(gt_list))
                gt_value = sum(gt_list)
                pred_value = sum(pred_list)
                print("GT: %d, predict: %d, rate:%f" % (gt_value, pred_value, (pred_value / gt_value)))
                # print("%f" % ((pred_value / gt_value)))
                print(pred_file_idx)
                # print(file_list)
                total_rate.append((pred_value / gt_value))

                subject = int(idx_spilt[0])
                gt_list = []
                pred_list = []
                file_list = []
                pred_file_idx = []

            gt_mask = cv2.imread(os.path.join(gt_path, idx), cv2.IMREAD_GRAYSCALE)
            pred_mask = cv2.imread(os.path.join(pred_path, idx), cv2.IMREAD_GRAYSCALE)
            gt_unique = int(len(np.unique(gt_mask))) - 1
            pred_unique = int(len(np.unique(pred_mask))) - 1

            gt_list.append(gt_unique)
            pred_list.append(pred_unique)

            if pred_unique == 1:
                int_idx = int(idx_spilt[1].split('.')[0])
                pred_file_idx.append(int_idx)

            if gt_unique != pred_unique:
                file_list.append(idx_spilt[1])

        # Printing last subject
        print("%d" % (int(idx_spilt[0])))
        gt_value = sum(gt_list)
        pred_value = sum(pred_list)
        print("GT: %d, predict: %d, rate:%f" % (gt_value, pred_value, (pred_value/gt_value)))
        # print("%f" % ((pred_value / gt_value)))
        # print(file_list)
        print(pred_file_idx)
        total_rate.append((pred_value / gt_value))

        # total_rate > 1.0 (GT<predict)
        print("\n Total Rate=%f"%(np.mean(total_rate)))


if __name__ == '__main__':

    # model_path_name = "0819_update_all"
    # model_path_name = "0819_update_all_FD"
    # model_path_name = "0819_update_all_rpn_1"
    # model_path_name = "0819_update_all_rpn_2"
    # model_path_name = "0819_update_all_rpn_3"
    # model_path_name = "0823_fd_rpn"

    # model_path_name = "0824_fd_rpn_1_1"
    # model_path_name = "0824_fd_rpn_2_1"
    # model_path_name = "0824_fd_rpn_3_1"

    # model_path_name = "0824_fd_rpn_1_0.5"
    # model_path_name = "0824_fd_rpn_2_0.5"
    # model_path_name = "0824_fd_rpn_3_0.5"

    # model_path_name = "maskunet_28_default"
    model_path_name = "maskrcnn_default"

    gpu_idx = 2
    train_batch_size = 2

    print("*"*50)
    print("Model_Path : " + model_path_name)
    print("Batch size: " + str(train_batch_size))
    print("*" * 50)

    main('train', model_path_name, gpu_idx, train_batch_size)
    # main('test', model_path_name, gpu_idx)
    # main('detection', model_path_name)
