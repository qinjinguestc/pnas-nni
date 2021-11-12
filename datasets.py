# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
from glob import glob
import re
import time

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    Scale, ToTensor)
import SimpleITK as sitk
from PIL import Image


mha_result = './mha result/{}/'.format(time.strftime('%Y%m%d-%H%M%S'))


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def get_dataset(cls, cutout_length=0, test=None):
    if cls == "cifar10":
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]

        transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]
        cutout = []
        if cutout_length > 0:
            cutout.append(Cutout(cutout_length))

        train_transform = transforms.Compose(transf + normalize + cutout)
        valid_transform = transforms.Compose(normalize)

        dataset_train = CIFAR10(root="D:/Code/data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="D:/Code/data", train=False, download=True, transform=valid_transform)

    elif cls == 'brats2015':
        MEAN = [0.2007713, 0.28985304, 0.31852704, 0.36998123]
        STD = [0.59269184, 0.9364896, 1.0523965, 1.1754125]

        transf = [
            transforms.RandomHorizontalFlip()
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]
        cutout = []
        if cutout_length > 0:
            cutout.append(Cutout(cutout_length))

        train_transform = transforms.Compose(transf + normalize + cutout)
        valid_transform = transforms.Compose(normalize)

        # datapath = '/home/soap/code/data'
        datapath = 'D:/Code/data'
        dataset_train = BraTS2015(root=datapath + '/BRATS2015_slice_all', mode='train', transform=train_transform)
        dataset_valid = BraTS2015(root=datapath + '/BRATS2015_slice_all', mode='val', transform=valid_transform)
        dataset_test = BraTS2015(root=datapath + '/BRATS2015_slice_all', mode='test', transform=valid_transform)

    else:
        raise NotImplementedError
    if test is True:
        return dataset_test
    else:
        return dataset_train, dataset_valid


class BraTS2015(data.Dataset):

    def __init__(self, root, mode='train', transform=None):
        self.mode = mode
        self.bra_path = []
        self.OT_path = []
        self.Flair_path = []
        self.T1_path = []
        self.T1c_path = []
        self.T2_path = []
        self.transform = transform

        if mode != 'test':
            for LorH in os.listdir(root+'/Training'):
                path = os.path.join(root, 'Training', LorH)
                for filename in os.listdir(path):
                    braPath = os.path.join(path, filename)
                    self.bra_path.append(braPath)
            random.shuffle(self.bra_path)

        elif mode == 'test':
            path = os.path.join(root, 'Testing')
            for filename in os.listdir(path):
                braPath = os.path.join(path, filename)
                self.bra_path.append(braPath)


        num_dataset = len(self.bra_path)
        split = int(num_dataset * 0.8)
        train_path = self.bra_path[0:split]
        val_path = self.bra_path[split:num_dataset]
        test_path = self.bra_path

        if mode == 'train':
            self.T1_path, self.T1c_path, self.Flair_path, self.T2_path, self.OT_path = self.read_data(train_path)
        elif mode == 'val':
            self.T1_path, self.T1c_path, self.Flair_path, self.T2_path, self.OT_path = self.read_data(val_path)
        elif mode == 'test':
            self.T1_path, self.T1c_path, self.Flair_path, self.T2_path, self.OT_path = self.read_data(test_path)

    def __getitem__(self, index):

        if self.mode == "train":
            ot = self.OT_path[index]
            t1 = self.T1_path[index]
            t2 = self.T2_path[index]
            t1c = self.T1c_path[index]
            flair = self.Flair_path[index]
        elif self.mode == 'val':
            ot = self.OT_path[index]
            t1 = self.T1_path[index]
            t2 = self.T2_path[index]
            t1c = self.T1c_path[index]
            flair = self.Flair_path[index]
        elif self.mode == 'test':
            ot = np.zeros((240, 240))
            t1 = self.T1_path[index]
            t2 = self.T2_path[index]
            t1c = self.T1c_path[index]
            flair = self.Flair_path[index]

        label = self.read_label(ot)
        image_array_t1 = self.read_image(t1)
        image_array_t2 = self.read_image(t2)
        image_array_t1c = self.read_image(t1c)
        image_array_flair = self.read_image(flair)
        image_array = np.concatenate((image_array_flair, image_array_t1, image_array_t1c, image_array_t2), axis=-1)

        if self.transform is not None:
            if self.mode != 'test':
                image = Image.fromarray(image_array.astype(np.uint8))
                image = self.transform(image)
                label = ToTensor()(label)
                label = label.long().squeeze()
                return image, label
            elif self.mode == 'test':
                image = Image.fromarray(image_array.astype(np.uint8))
                image = self.transform(image)
                label = ToTensor()(label)
                return image, label

    def __len__(self):
        return len(self.Flair_path)

    def saveItk(self, array):
        array = np.asarray(array)
        if self.saveArr is not None:
            self.saveArr = np.concatenate([self.saveArr, array], axis=0)
        else:
            self.saveArr = array
        if self.test_batch_index == 0:
            # array[array >= 230] = 255  # 4
            # array[array <= 2] = 254  # 0
            # array[array <= 64] = 253  # attention
            # array[array <= 128] = 252  # 2
            # array[array < 230] = 251  # 3
            # array[array == 255] = 4
            # array[array == 254] = 0
            # array[array == 253] = attention
            # array[array == 252] = 2
            # array[array == 251] = 3
            img = sitk.GetImageFromArray(self.saveArr)
            path = self.Flair_path[self.test_batch_count - 2]
            name = 'VSD.DENSEUNET_test' + re.findall(r"\.\d+\.", path)[0] + 'mha'
            print(name, self.test_batch_count-1)
            if not os.path.exists(mha_result):
                os.makedirs(mha_result)
            sitk.WriteImage(sitk.Cast(img, sitk.sitkInt8), os.path.join(mha_result, name), True)
            self.saveArr = None

    @staticmethod
    def read_data(brapath):

        T1_path = []
        T1c_path = []
        Flair_path = []
        T2_path = []
        OT_path = []

        for path in brapath:
            T1_path = T1_path + sorted(glob(path + '/*T1.*/*.gz'), key=lambda name: name[-11:-7])
            T1c_path = T1c_path + sorted(glob(path + '/*T1c.*/*.gz'), key=lambda name: name[-11:-7])
            Flair_path = Flair_path + sorted(glob(path + '/*Flair.*/*.gz'), key=lambda name: name[-11:-7])
            T2_path = T2_path + sorted(glob(path + '/*T2.*/*.gz'), key=lambda name: name[-11:-7])
            OT_path = OT_path + sorted(glob(path + '/*OT.*/*.gz'), key=lambda name: name[-11:-7])

        return T1_path, T1c_path, Flair_path, T2_path, OT_path

    @staticmethod
    def input_transform():
        return Compose([
            ToTensor(),
            # flair,t1,t1c,t2
            Normalize(
                mean=[0.2007713, 0.28985304, 0.31852704, 0.36998123],
                std=[0.59269184, 0.9364896, 1.0523965, 1.1754125]
            ),
            # ToTensor()
        ])

    @staticmethod
    def read_image(file_name, format=None):
        mha = sitk.ReadImage(file_name)
        spacing = mha.GetSpacing()
        image = sitk.GetArrayFromImage(mha)
        image = np.expand_dims(image, -1)
        return image

    @staticmethod
    def read_label(file_name, dtype='float'):
        if type(file_name) is np.ndarray:
            return file_name
        else:
            # In some cases, `uint8` is not enough for label
            mha = sitk.ReadImage(file_name)
            image = sitk.GetArrayFromImage(mha)
            # image = np.expand_dims(image, -attention)
            return np.asarray(image, dtype=dtype)
