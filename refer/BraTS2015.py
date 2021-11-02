import os
from glob import glob

import SimpleITK as sitk
import numpy as np
import random
import re
import shutil
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor, Scale, Resize
from .base import  BaseDataset


mha_save_path = r"./mha_save_path/"
mha_ground_truth = mha_save_path + "mha_ground_truth/"
mha_result = mha_save_path + "mha_result/"


class BraTS2015(BaseDataset):
    IN_CHANNELS = 4
    NUM_CLASS = 5
    CROP_SIZE = 256
    CLASS_WEIGHTS = None

    def __init__(self, root, split='train', mode='train'):
        basePath = r'/home/soap/code/data/BRATS2015_slice_all'
        self.mode = mode
        self.bra_path = []
        self.OT_path = []
        self.Flair_path = []
        self.T1_path = []
        self.T1c_path = []
        self.T2_path = []

        for LorH in os.listdir(basePath + '/Training'):
            path = os.path.join(basePath, 'Training', LorH)
            for filename in os.listdir(path):
                braPath = os.path.join(path, filename)
                self.bra_path.append(braPath)

        random.shuffle(self.bra_path)
        num_dataset = len(self.bra_path)
        split = int(num_dataset * 0.8)
        train_path = self.bra_path[0:split]
        val_path = self.bra_path[split:num_dataset]
        test_path = self.bra_path[0:5]

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
            ot = self.OT_path[index]
            t1 = self.T1_path[index]
            t2 = self.T2_path[index]
            t1c = self.T1c_path[index]
            flair = self.Flair_path[index]

        label = self.read_label(ot)
        image_array_t1 = self.read_image(t1)
        image_array_t2 = self.read_image(t2)
        image_array_t1c = self.read_image(t1c)
        image_array_flair = self.read_image(flair)
        image = np.concatenate((image_array_flair, image_array_t1, image_array_t1c, image_array_t2), axis=-1)

        image = self.input_transform()(image)
        label = ToTensor()(label)

        return image, label
        
    def __len__(self):
        return len(self.OT_path)

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
        image = sitk.GetArrayFromImage(mha)
        image = np.expand_dims(image, -1)
        return image

    @staticmethod
    def read_label(file_name, dtype='float'):
        # In some cases, `uint8` is not enough for label
        mha = sitk.ReadImage(file_name)
        image = sitk.GetArrayFromImage(mha)
        # image = np.expand_dims(image, -attention)
        return np.asarray(image, dtype=dtype)

    def read_data(self, brapath):

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

