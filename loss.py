# from tensorflow.keras import backend as K
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt



# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5:
        return [1, 2, 3]
    # Two dimensional
    elif len(shape) == 4:
        return [1, 2]
    # Exception - Unknown
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


################################
#           tp_fp_fn          #
################################

def tp_fp_fn(y_true, y_pred):

    tp = torch.sum(y_true * y_pred)
    fp = torch.sum((1-y_true) * y_pred)
    fn = torch.sum(y_true * (1-y_pred))

    return tp, fp, fn


################################
#           Dice loss          #
################################
def dice_loss(delta=0.5, smooth=0.000001):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def loss_function(y_true, y_pred):
        tp, fp, fn = tp_fp_fn(y_true, y_pred)
        # Calculate Dice score
        dice_coefficient = (2 * tp + smooth) / (2 * tp + fn + fp + smooth)
        # Sum up classes to one score
        dice_loss = 1 - dice_coefficient
        return dice_loss

    return loss_function


################################
#         Tversky loss         #
################################
def tversky_loss(delta=0.7, smooth=0.000001):
    """Tversky loss function for image segmentation using 3D fully convolutional deep networks
	Link: https://arxiv.org/abs/1706.05721
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def loss_function(y_true, y_pred):
        tp, fp, fn = tp_fp_fn(y_true, y_pred)
        # Calculate Dice score
        tversky_coeffi = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Sum up classes to one score
        tversky_loss = 1 - tversky_coeffi
        return tversky_loss

    return loss_function


################################
#       Dice coefficient       #
################################
def dice_coefficient(delta=0.5, smooth=0.000001):
    """The Dice similarity coefficient, also known as the Sørensen–Dice index or simply Dice coefficient, is a statistical tool which measures the similarity between two sets of data.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Sum up classes to one score
        dice = K.sum(dice_class, axis=[-1])
        # adjusts loss to account for number of classes
        num_classes = K.cast(K.shape(y_true)[-1], 'float32')
        dice = dice / num_classes
        return dice

    return loss_function


################################
#          Combo loss          #
################################
def combo_loss(alpha=0.5, beta=0.5):
    """Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
    Link: https://arxiv.org/abs/1805.02798
    Parameters
    ----------
    alpha : float, optional
        controls weighting of dice and cross-entropy loss., by default 0.5
    beta : float, optional
        beta > 0.5 penalises false negatives more than false positives., by default 0.5
    """

    def loss_function(y_true, y_pred):
        dice = dice_coefficient()(y_true, y_pred)
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if beta is not None:
            beta_weight = np.array([beta, 1 - beta])
            cross_entropy = beta_weight * cross_entropy
        # sum over classes
        cross_entropy = K.mean(K.sum(cross_entropy, axis=[-1]))
        if alpha is not None:
            combo_loss = (alpha * cross_entropy) - ((1 - alpha) * dice)
        else:
            combo_loss = cross_entropy - dice
        return combo_loss

    return loss_function


################################
#      Focal Tversky loss      #
################################
def focal_tversky_loss(delta=0.7, gamma=0.75, smooth=0.000001):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """

    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Sum up classes to one score
        focal_tversky_loss = K.sum(K.pow((1 - tversky_class), gamma), axis=[-1])
        # adjusts loss to account for number of classes
        num_classes = K.cast(K.shape(y_true)[-1], 'float32')
        focal_tversky_loss = focal_tversky_loss / num_classes
        return focal_tversky_loss

    return loss_function


################################
#          Focal loss          #
################################
def focal_loss(alpha=None, beta=None, gamma_f=2.):
    """Focal loss is used to address the issue of the class imbalance problem. A modulation term applied to the Cross-Entropy loss function.
    Parameters
    ----------
    alpha : float, optional
        controls weight given to each class, by default None
    beta : float, optional
        controls relative weight of false positives and false negatives. Beta > 0.5 penalises false negatives more than false positives, by default None
    gamma_f : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 2.
    """

    def loss_function(y_true, y_pred):
        num_tensor = 1
        for idx in y_pred.size():
            num_tensor = num_tensor * idx

        # Clip values to prevent division by zero error
        epsilon = epsilon = 1e-07
        y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)

        fore_ce = -y_true * torch.log(y_pred)
        back_ce = -(1 - y_true) * torch.log(1 - y_pred)

        fore_focal = torch.pow(1-y_pred, gamma_f) * fore_ce
        fore_focal = torch.sum(fore_focal) / num_tensor

        back_focal = torch.pow(1-(1-y_pred),gamma_f) * back_ce
        back_focal = torch.sum(back_focal) / num_tensor

        focal_loss = fore_focal + back_focal
        return focal_loss

    return loss_function


################################
#       Hybrid Focal loss      #
################################
def hybrid_focal_loss(weight=None, alpha=None, beta=None, gamma=0.75, gamma_f=2.):
    """Default is the linear unweighted sum of the Focal loss and Focal Tversky loss
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to Focal Tversky loss and Focal loss, by default None
    alpha : float, optional
        controls weight given to each class, by default None
    beta : float, optional
        controls relative weight of false positives and false negatives. Beta > 0.5 penalises  false negatives more than false positives, by default None
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 0.75
    gamma_f : float, optional
        Focal loss' focal parameter controls degree of down-weighting of easy examples, by default 2.
    """

    def loss_function(y_true, y_pred):
        # Obtain Focal Dice loss
        focal_tversky = focal_tversky_loss(gamma=gamma)(y_true, y_pred)
        # Obtain Focal loss
        focal = focal_loss(alpha=alpha, beta=beta, gamma_f=gamma_f)(y_true, y_pred)
        # return weighted sum of Focal loss and Focal Dice loss
        if weight is not None:
            return (weight * focal_tversky) + ((1 - weight) * focal)
        else:
            return focal_tversky + focal

    return loss_function


################################
#     Asymmetric Focal loss    #
################################
def asymmetric_focal_loss(delta=0.25, gamma=2.):
    def loss_function(y_true, y_pred):
        """For Imbalanced datasets
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.25
        gamma : float, optional
            Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
        """


        epsilon = 1e-07
        y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)

        num_tensor=1
        for idx in y_pred.size():
            num_tensor = num_tensor * idx

        fore_ce = -y_true * torch.log(y_pred)
        back_ce = -(1-y_true) * torch.log(1-y_pred)

        # version 1
        # fore_ce = delta * fore_ce
        fore_ce = torch.sum(fore_ce) / num_tensor

        # calculate losses separately for each class, only suppressing background class
        # version 1
        # back_ce = torch.pow(y_pred, gamma) * back_ce
        # back_ce = (1 - delta) * back_ce
        back_ce = torch.sum(back_ce) / num_tensor

        loss = fore_ce + back_ce

        return loss
    return loss_function



#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75, smooth=0.000001):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    """

    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = 1e-07
        y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)

        fore_tversky = tversky_loss(delta=delta)(y_true, y_pred)
        back_tversky = tversky_loss(delta=(1-delta))((1-y_true), (1-y_pred))

        # calculate losses separately for each class, only enhancing foreground class
        #back_dice = (1 - dice_class[:, 0])
        #fore_dice = (1 - dice_class[:, 1]) * K.pow(1 - dice_class[:, 1], -gamma)

        # version 1
        # fore_tversky = fore_tversky * torch.pow(fore_tversky, -gamma)

        loss = back_tversky + fore_tversky

        return loss

    return loss_function


################################
#      Unified Focal loss      #
################################
def unified_focal_loss(weight=0.5, delta=0.6, gamma=0.2):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to Asymmetric Focal Tversky loss and Asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.2
    """

    def loss_function(y_true, y_pred):
        asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true, y_pred)
        # Obtain Asymmetric Focal loss
        asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true, y_pred)
        # return weighted sum of Asymmetrical Focal loss and Asymmetric Focal Tversky loss
        if weight is not None:
            return (weight * asymmetric_ftl) + ((1 - weight) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl

    return loss_function



class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()

    # def forward(self, gt, predict, smooth=1.):
    def forward(self, gt, predict, smooth=0.000001):
        #gt one-hot encoding
        # gt[gt>0.5] = 1
        # gt[gt<=0.5] = 0

        intersection = (gt * predict).sum()

        dice_coefficient = (2*intersection+ smooth) / (gt.sum() + predict.sum() + smooth)

        loss_dice = (1-dice_coefficient)

        return loss_dice

class weightDiceLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(weightDiceLoss, self).__init__()

    # def forward(self, gt, predict, smooth=1.):
    def forward(self, gt, predict, v1=0.15, smooth=0.000001):

        # #gt one-hot encoding
        # gt[gt>0.5] = 1
        # gt[gt<=0.5] = 0


        # gt_modi = 2*gt - 1
        # predict_modi = 2*predict - 1

        v2 = 1-v1

        weight = (v2-v1)*gt + v1

        gt = weight*gt
        predict = weight*predict

        fp = (1 - gt) * predict

        intersection = (gt*predict).sum()

        dice_coefficient = (2*intersection + smooth) / ((gt*gt).sum() + (predict*predict).sum() + smooth)

        loss_dice = (1-dice_coefficient)

        fp_coefficient = (intersection + smooth) / ((gt*gt).sum() + (fp*fp).sum() + smooth)
        loss_fp = 1-fp_coefficient

        loss_total = loss_dice + loss_fp

        # loss_log_dice = -torch.log(dice_coefficient)

        return loss_total

class logDiceLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(logDiceLoss, self).__init__()

    def forward(self, gt, predict, smooth=0.000001):

        # gt = one_hot encoding
        # predict = network output
        intersection = (gt * predict).sum()

        dice_coefficient = (2*intersection + smooth) / (gt.sum() + predict.sum() + smooth)

        loss_dice = (1-dice_coefficient)

        loss_total = torch.log((torch.exp(loss_dice) + torch.exp(-loss_dice)) / 2.0)

        return loss_total



class ComboLoss(torch.nn.Module):
    def __init__(self):
        super(ComboLoss, self).__init__()

    def forward(self, gt, predict_sig, smooth=1):
        ALPHA = 0.7  # < 0.5 penalises FP more, > 0.5 penalises FN more
        CE_RATIO = 0.7  # weighted contribution of modified CE loss compared to Dice loss
        eps = 1e-07
        predict_sig = predict_sig.view(-1)
        gt = gt.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (predict_sig * gt).sum()
        dice = (2. * intersection + smooth) / (predict_sig.sum() + gt.sum() + smooth)
        # log_dice_loss = torch.log((torch.exp(1-dice) + torch.exp(-(1-dice))) / 2.0)

        predict_sig = torch.clamp(predict_sig, eps, 1.0 - eps)
        out = - ((ALPHA * ((gt * torch.log(predict_sig)) + ((1 - ALPHA) * (1.0 - gt) * torch.log(1.0 - predict_sig)))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) + ((1 - CE_RATIO) * (1-dice))
        # combo = (CE_RATIO * weighted_ce) + ((1 - CE_RATIO) * (log_dice_loss*20))

        return combo



class TverLoss(torch.nn.Module):
    def __init__(self):
        super(TverLoss, self).__init__()

    def forward(self, gt, predict, delta=0.60, smooth=0.000001):

        # gt = one_hot encoding
        # predict = network output

        tp = torch.sum(gt * predict)
        fp = torch.sum((1 - gt) * predict)
        fn = torch.sum(gt * (1 - predict))

        tversky_coeffi = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)

        tversky_loss = 1 - tversky_coeffi

        loss_BCE = F.binary_cross_entropy(predict, gt)

        loss_total = tversky_loss + loss_BCE
        return loss_total


class asymTverLoss(torch.nn.Module):
    def __init__(self):
        super(asymTverLoss, self).__init__()

    def forward(self, gt, predict, beta=1.5, smooth=1.):

        # gt = one_hot encoding
        # predict = network output

        tp = torch.sum(gt * predict)
        fp = torch.sum((1 - gt) * predict)
        fn = torch.sum(gt * (1 - predict))

        weight = (beta**2) / (1+beta**2)

        asym = tp / (tp + weight*fn + (1-weight)*fp + smooth)

        tversky_loss = 1 - asym

        # loss_BCE = F.binary_cross_entropy(predict, gt)

        return tversky_loss


class focalDiceLoss(torch.nn.Module):
    def __init__(self):
        super(focalDiceLoss, self).__init__()

    def forward(self, gt, predict, smooth=0.000001):

        # gt = one_hot encoding
        # predict = network output

        tp = gt * predict
        fp = (1 - gt) * predict
        fn = gt * (1 - predict)

        loss_focal_1 = 1 - (((2*tp).sum() + smooth) / ((tp+fn+fp).sum() + (gt.sum()) + smooth))

        loss_focal_2 = 1 - ((tp.sum() + smooth) / ((fn.sum()) + (gt.sum()) + smooth))

        loss_focal_dice = loss_focal_1 + loss_focal_2

        return loss_focal_dice


class BinaryFocalLoss(torch.nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=1.5, threshold=0.1, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # self.smooth = 1e-6
        self.smooth = 1e-4  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.threshold = threshold

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)
        # probability: simoid -> 0~1 / clamp -> side cut

        valid_mask = None
        if self.ignore_index is not None: # skip
            valid_mask = (target != self.ignore_index).float()

        # pos_mask = (target == 1).float()
        # neg_mask = (target == 0).float()

        # modified hyoseok
        pos_mask = (target >= self.threshold).float()
        neg_mask = (target < self.threshold).float()
        # pose_mask & neg_mask = divied target value

        if valid_mask is not None: # skip
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        # neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)
        neg_loss = -self.alpha * neg_weight * torch.log(1-prob)
        loss = pos_loss + neg_loss
        loss = loss.mean()

        return loss


class coshBinaryFocalLoss(torch.nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=1.5, threshold=0.1, smooth=0.000001, weight=0.5,dice_alpha=0.8,  ignore_index=None, reduction='mean', **kwargs):
        super(coshBinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # self.smooth = 1e-6
        self.smooth = 1e-4  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.threshold = threshold
        self.smooth = smooth
        self.weight = weight
        self.dice_alpha = dice_alpha

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)
        # probability: simoid -> 0~1 / clamp -> side cut

        valid_mask = None
        if self.ignore_index is not None: # skip
            valid_mask = (target != self.ignore_index).float()

        # modified hyoseok
        pos_mask = (target >= self.threshold).float()
        neg_mask = (target < self.threshold).float()
        # pose_mask & neg_mask = divied target value

        if valid_mask is not None: # skip
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * torch.log(1 - prob)

        ####################################################################################################
        # Dice Loss
        intersection = (pos_mask * prob).sum()
        dice_coefficient = (2*intersection+ self.smooth) / (pos_mask.sum() + prob.sum() + self.smooth)

        loss_foreground_dice = (1-dice_coefficient)

        # tp = torch.sum(pos_mask * prob)
        # fp = torch.sum((1 - pos_mask) * prob)
        # fn = torch.sum(pos_mask * (1 - prob))  # FN
        #
        # tversky_coefficient = tp / (tp + (self.dice_alpha * fp) + ((1-self.dice_alpha) * fn) + self.smooth)
        # loss_foreground_tversky = 1-tversky_coefficient
        ####################################################################################################

        loss = pos_loss + neg_loss
        loss = loss.mean()

        # focal loss + foreground dice loss function
        # loss = (self.weight*loss) + ((1-self.weight)*loss_foreground_dice)

        # focal loss + log dice loss function
        # loss = (self.weight*loss) + ((1-self.weight)*(-torch.log(dice_coefficient)))

        # focal loss + log tversky loss function
        # loss = (self.weight*loss) + ((1-self.weight)*-torch.log(tversky_coefficient))

        # focal loss + log cosh dice loss function
        loss = (self.weight*loss) + ((1-self.weight)*torch.log((torch.exp(loss_foreground_dice) + torch.exp(-loss_foreground_dice)) / 2.0))

        return loss