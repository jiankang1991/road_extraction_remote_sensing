from torch import nn

from torch.nn import functional as F

import torch
from torch.autograd import Variable

import lovasz_losses as L


# http://jeffwen.com/2018/02/23/road_extraction
class BCEDiceLoss(nn.Module):
    def __init__(self, penalty_weight=None, size_average=True):
        super().__init__()
        self.penalty_weight = penalty_weight
        # self.BCE_weight = BCE_weight
    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)
        
        # print(pred.size())

        # BCE loss
        # if self.BCE_weight is not None:
        #     bce_loss = nn.BCELoss(weight=self.BCE_weight)(pred, truth).double()
        #     print('using weighted BCE loss')
        # else:            
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2. * (pred * truth).double().sum() + 1) / (pred.double().sum() + truth.double().sum() + 1)
        
        if self.penalty_weight:
            # print('penalty weight is {}'.format(self.penalty_weight))
            dice_loss = self.penalty_weight * (1 - dice_coef)
        else:
            dice_loss = (1 - dice_coef)
        
        loss = bce_loss + dice_loss
        return loss, bce_loss, dice_loss

class BCELovaszLoss(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):

        pred = input.view(-1)
        truth = target.view(-1)

        bce_loss = nn.BCEWithLogitsLoss()(pred, truth).double()

        # lovasz loss
        lovasz_loss = L.lovasz_hinge(input, target, per_image=False)

        loss = bce_loss + lovasz_loss.double()

        return loss, bce_loss, lovasz_loss.double()


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
# https://github.com/ternaus/robot-surgery-segmentation/blob/master/evaluate.py
def jaccard_index(input, target):
    """IoU calculation """
    num_in_target = input.size(0)

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    # intersection = (pred*truth).long().sum(1).data.cpu()[0]
    intersection = (pred * truth).sum(1)

    # union = input.long().sum().data.cpu()[0] + target.long().sum().data.cpu()[0] - intersection
    union = pred.sum(1) + truth.sum(1) - intersection

    score = (intersection + 1e-15) / (union + 1e-15)

    return score.mean().data[0]



# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().data[0]

# https://github.com/ternaus/robot-surgery-segmentation/blob/master/loss.py
class LossBinaryJaccard(object):
    """
    Loss defined as BCE - log(soft_jaccard)
    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=None):
        # self.nll_loss = nn.BCEWithLogitsLoss()
        # self.BCE_weight = BCE_weight
        # if self.BCE_weight is not None:
        #     print('using weighted BCE loss')
        #     self.nll_loss = nn.BCELoss(self.BCE_weight)
        # else:
        self.nll_loss = nn.BCELoss()
        
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        BCE_loss = self.nll_loss(outputs, targets)

        # print('penalty weight is {}'.format(self.jaccard_weight))
        eps = 1e-15
        jaccard_target = (targets == 1).float()
        jaccard_output = F.sigmoid(outputs)

        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()

        Jaccard_loss = 0 - self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        loss = BCE_loss + Jaccard_loss
        # loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss, BCE_loss, Jaccard_loss


#### losses from http://blog.kaggle.com/2017/12/22/carvana-image-masking-first-place-interview/
#### https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/asanakoy/losses.py

class BCELoss2d(nn.Module):
    """
    Binary Cross Entropy loss function
    """
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)
        return self.bce_loss(logits_flat, labels_flat)

class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, labels):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score
    
##  http://geek.csdn.net/news/detail/126833
## https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208
class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        logits = logits.view(-1)
        gt = labels.view(-1)
        # http://geek.csdn.net/news/detail/126833
        loss = logits.clamp(min=0) - logits * gt + torch.log(1 + torch.exp(-logits.abs()))
        loss = loss * w
        loss = loss.sum() / w.sum()
        return loss


class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        w = weights.view(num, -1)
        w2 = w * w
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * ((w2 * intersection).sum(1) + 1) / (
            (w2 * m1).sum(1) + (w2 * m2).sum(1) + 1)
        score = 1 - score.sum() / num
        return score


class CombinedLoss(nn.Module):
    def __init__(self, is_weight=True, is_log_dice=False):
        super(CombinedLoss, self).__init__()
        self.is_weight = is_weight
        self.is_log_dice = is_log_dice
        if self.is_weight:
            self.weighted_bce = WeightedBCELoss2d()
            self.soft_weighted_dice = WeightedSoftDiceLoss()
        else:
            self.bce = BCELoss2d()
            self.soft_dice = SoftDiceLoss()

    def forward(self, logits, labels):
        size = logits.size()
        assert size[1] == 1, size
        logits = logits.view(size[0], size[2], size[3])
        labels = labels.view(size[0], size[2], size[3])
        if self.is_weight:
            batch_size, H, W = labels.size()
            if H == 128:
                kernel_size = 11
            elif H == 256:
                kernel_size = 11
            elif H == 512:
                kernel_size = 21
            elif H == 1024:
                kernel_size = 41
            elif H == 1280:
                kernel_size = 51
            else:
                raise ValueError('Unknown height')

            a = F.avg_pool2d(labels, kernel_size=kernel_size, padding=kernel_size // 2,
                             stride=1)
            ind = a.ge(0.01) * a.le(0.99)
            ind = ind.float()
            weights = Variable(torch.tensor.torch.ones(a.size())).cuda()

            w0 = weights.sum()
            weights += ind * 2
            w1 = weights.sum()
            weights = weights / w1 * w0

            bce_loss = self.weighted_bce(logits, labels, weights)
            dice_loss = self.soft_weighted_dice(logits, labels, weights)
        else:
            bce_loss = self.bce(logits, labels)
            dice_loss = self.soft_dice(logits, labels)

        if self.is_log_dice:
            l = bce_loss - (1 - dice_loss).log()
        else:
            l = bce_loss + dice_loss
        return l, bce_loss, dice_loss





