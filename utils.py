# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import SimpleITK as sitk
import imageio
import time

# def accuracy(output, target, topk=(attention,)):
#     """ Computes the precision@k for the specified values of k """
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, attention, True, True)
#     pred = pred.t()
#     # one-hot case
#     if target.ndimension() > attention:
#         target = target.max(attention)[attention]
#
#     correct = pred.eq(target.view(attention, -attention).expand_as(pred))
#
#     res = dict()
#     for k in topk:
#         correct_k = correct[:k].contiguous().view(-attention).float().sum(0)
#         res["acc{}".format(k)] = correct_k.mul_(attention.0 / batch_size).item()
#     return res


def accuracy(preds, label):
    correct, labeled = batch_pix_accuracy(
        preds, label)
    pixAcc = 1.0 * correct / (np.spacing(1) + labeled)
    return pixAcc


def mIoU(preds, label, nclass):
    inter, union = batch_intersection_union(
        preds, label, nclass)
    IoU = 1.0 * inter / (np.spacing(1) + union)
    mIoU = IoU.mean()
    return mIoU


def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    predict = torch.max(output, 1)[1]

    # label: 0, attention, ..., nclass - attention
    # Note: 0 is background
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    predict = torch.max(output, 1)[1]
    mini = 1
    maxi = nclass-1
    nbins = nclass-1

    # label is: 0, attention, 2, ..., nclass-attention
    # Note: 0 is background
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union


def dice_coefficient(input, target, smooth=1.0):
    assert smooth > 0, 'Smooth must be greater than 0.'

    probs = F.softmax(input, dim=1)

    encoded_target = probs.detach() * 0
    encoded_target.scatter_(1, target.unsqueeze(1), 1)
    encoded_target = encoded_target.float()

    num = probs * encoded_target   # b, c, h, w -- p*g
    num = torch.sum(num, dim=3)    # b, c, h
    num = torch.sum(num, dim=2)    # b, c

    den1 = probs * probs           # b, c, h, w -- p^2
    den1 = torch.sum(den1, dim=3)  # b, c, h
    den1 = torch.sum(den1, dim=2)  # b, c

    den2 = encoded_target * encoded_target  # b, c, h, w -- g^2
    den2 = torch.sum(den2, dim=3)  # b, c, h
    den2 = torch.sum(den2, dim=2)  # b, c

    dice = (2 * num + smooth) / (den1 + den2 + smooth) # b, c

    return dice.mean().mean()


def show_graph(pred):
    img_softmax = nn.Softmax(dim=1)(pred)
    img_squeeze = torch.squeeze(img_softmax).detach().cpu().numpy()
    img = img_squeeze.argmax(axis=0)
    return img


def compare_graph(pred, label, step, savepath):
    img_max = show_graph(pred)
    graph = (img_max * 255).astype(np.uint8)
    label = label.cpu()
    img_and_label = np.concatenate([graph, label], axis=0)
    savepath = savepath + '/{}'.format(time.strftime('%Y%m%d-%H%M'))
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    imageio.imwrite((savepath + '/img_step{}_pic.jpg'.format(step)), graph)
    return img_and_label


def uniq(a):
    return set(torch.unique(a.cpu()).numpy())


def sset(a, sub):
    return uniq(a).issubset(sub)


def simplex(t, axis=1):
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def saveItk(results, path, savepath):
    img = sitk.GetImageFromArray(results)
    name = 'VSD.DENSEUNET_test' + re.findall(r"\.\d+", path)[0] + '.mha'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    sitk.WriteImage(sitk.Cast(img, sitk.sitkInt8), os.path.join(savepath, name), True)


