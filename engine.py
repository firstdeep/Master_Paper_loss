import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn
from torchvision.models.detection import roi_heads

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
import numpy as np
from PIL import Image
import cv2


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # number of model parameters
        model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    total_loss = []

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        outputs = model(image)

        if (list(outputs[0]['boxes'].shape)[0] == 0):
            mask = np.zeros((256, 256), dtype=np.uint8)
        else:
            mask = Image.fromarray(outputs[0]['masks'][0, 0].mul(255).byte().cpu().numpy())

        img_mask = np.array(mask)
        img_mask[img_mask > 127] = 255
        img_mask[img_mask <= 127] = 0
        img_mask = (img_mask/255).astype(np.uint32)

        target_mask = Image.fromarray(targets[0]['masks'][0].byte().cpu().numpy())
        target_mask = np.array(target_mask).astype(np.uint32)

        p_sum = img_mask.sum()
        t_sum = target_mask.sum()

        intersection = np.bitwise_and(img_mask, target_mask).sum()
        union = np.bitwise_or(img_mask, target_mask).sum()

        jaccard = intersection / union
        dice = 2.0*intersection / (p_sum + t_sum)

        total_loss.append(((jaccard+dice)/2.0))

    total_loss = np.array(total_loss)
    return total_loss.mean()
