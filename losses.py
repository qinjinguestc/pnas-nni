from typing import List

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt

from torch import Tensor, einsum

from utils import simplex, one_hot


def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


def compute_edts_forPenalizedLoss(GT):
    """
    GT.shape = (batch_size, x,y,z)
    only for binary segmentation
    """
    res = np.zeros(GT.shape)
    for i in range(GT.shape[0]):
        posmask = GT[i]
        negmask = ~posmask
        pos_edt = distance_transform_edt(posmask)
        pos_edt = (np.max(pos_edt) - pos_edt) * posmask
        neg_edt = distance_transform_edt(negmask)
        neg_edt = (np.max(neg_edt) - neg_edt) * negmask

        res[i] = pos_edt / np.max(pos_edt) + neg_edt / np.max(neg_edt)
    return res


class DistBinaryDiceLoss(nn.Module):
    """
    Distance map penalized Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation
    """

    def __init__(self, smooth=1e-5):
        super(DistBinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, net_output, gt):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # one hot code for gt
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        gt_temp = gt[:, 0, ...].type(torch.float32)
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(gt_temp.cpu().numpy() > 0.5) + 1.0
        # print('dist.shape: ', dist.shape)
        dist = torch.from_numpy(dist)

        if dist.device != net_output.device:
            dist = dist.to(net_output.device).type(torch.float32)

        tp = net_output * y_onehot
        tp = torch.sum(tp[:, 1, ...] * dist, (1, 2, 3))

        dc = (2 * tp + self.smooth) / (torch.sum(net_output[:, 1, ...], (1, 2, 3)) + torch.sum(y_onehot[:, 1, ...],
                                                                                               (1, 2, 3)) + self.smooth)

        dc = dc.mean()

        return -dc

class BDLoss(nn.Module):
    def __init__(self):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        # self.do_bg = do_bg

    def forward(self, net_output, target, bound):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # print('net_output shape: ', net_output.shape)
        pc = net_output[:, 1:, ...].type(torch.float32)
        dc = bound[:,1:, ...].type(torch.float32)

        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", pc, dc)
        bd_loss = multipled.mean()

        return bd_loss

