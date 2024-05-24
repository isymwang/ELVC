import os

import logging
import cv2
from PIL import Image
import imageio
import numpy as np
import torch
import torch.utils.data as data
from os.path import join, exists
import math
import random
import sys
import json
import random
# from subnet.basics import *
from fvc_net.ms_ssim_torch import ms_ssim
from augmentation import random_flip, random_crop_and_pad_image_and_labels
from fvc_net.basic import *



trainpath='D:/wangyiming/data/train_data'

testpath= 'G:/wangyiming/data/test_data'





class HEVCDataSet(data.Dataset):
    def __init__(self, root="", filelist="G:/wangyiming/data/test_data/HEVC_filelists/B.txt", testfull=True):
        with open(filelist) as f:
            folders = f.readlines()
        self.input = []
        self.hevcclass = []
        for folder in folders:
            seq = folder.rstrip()
            # imlist = os.listdir(os.path.join(root, seq))
            cnt = 100
            gop=10

            if testfull:
                framerange = cnt // gop
            else:
                framerange = 1
            for i in range(framerange):
                inputpath = []
                for j in range(gop):
                    inputpath.append(os.path.join(root, seq, 'BT.601','im' + str(i * 10 + j + 1).zfill(5) + '.png'))

                self.input.append(inputpath)


    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_images = []
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)).astype(np.float32) / 255.0  # [3, h, w]

            input_images.append(input_image)

        input_images = np.array(input_images)
        return input_images




