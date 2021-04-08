import logging
from glob import glob
from os import listdir, path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.cityscapes.labels import id2trainid

sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from config import *


class CityscapesDataset(Dataset):
    def __init__(self, type='train', scale=1, extra=False):
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        img_suffix = '_leftImg8bit'
        mask_fine_suffix = '_gtFine_labelIds'
        mask_coarse_suffix = '_gtCoarse_labelIds'
        self.type = type
        self.data = []
        if type == 'train':
            for city in listdir(dir_imgs_train):
                city_dir = path.join(dir_imgs_train, city)
                for file in listdir(city_dir):
                    if file.startswith('.'):
                        continue
                    filename = file.rsplit(sep='_', maxsplit=1)[0]
                    img_file = path.join(dir_imgs_train, city, f'{filename}{img_suffix}.png')
                    mask_file = path.join(dir_masks_train, city, f'{filename}{mask_fine_suffix}.png')
                    self.data.append((img_file, mask_file))
            if extra:
                for city in listdir(dir_extra_imgs_train):
                    city_dir = path.join(dir_extra_imgs_train, city)
                    for file in listdir(city_dir):
                        if file.startswith('.'):
                            continue
                        filename = file.rsplit(sep='_', maxsplit=1)[0]
                        img_file = path.join(dir_extra_imgs_train, city, f'{filename}{img_suffix}.png')
                        mask_file = path.join(dir_extra_masks_train, city, f'{filename}{mask_coarse_suffix}.png')
                        self.data.append((img_file, mask_file))
        elif type == 'val':
            for city in listdir(dir_imgs_val):
                city_dir = path.join(dir_imgs_val, city)
                for file in listdir(city_dir):
                    if file.startswith('.'):
                        continue
                    filename = file.rsplit(sep='_', maxsplit=1)[0]
                    img_file = path.join(dir_imgs_val, city, f'{filename}{img_suffix}.png')
                    mask_file = path.join(dir_masks_val, city, f'{filename}{mask_fine_suffix}.png')
                    self.data.append((img_file, mask_file))

        logging.info(f'Creating dataset with {len(self.data)} examples')


    def __len__(self):
        return len(self.data)


    def resize(self, img_arr: np.ndarray):
        return cv2.resize(img_arr, None,
                          fx=self.scale, fy=self.scale,
                          interpolation=cv2.INTER_NEAREST)


    def __getitem__(self, i):
        img_file, mask_file = self.data[i]
        # get img array
        img = cv2.imread(img_file, flags=cv2.IMREAD_COLOR)  # shape (h, w, 3)
        mask = cv2.imread(mask_file, flags=cv2.IMREAD_GRAYSCALE)  # shape (h, w)
        height, width = mask.shape
        for y in range(height):
            for x in range(width):
                mask[y][x] = id2trainid[mask[y][x]]

        # resize & to tensor
        new_size = tuple(round(size * self.scale) for size in img.shape[0:2])
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(new_size)
        ])
        img_tensor = tf(img)

        mask = self.resize(mask)
        mask_tensor = torch.from_numpy(mask).type(torch.ByteTensor)

        return {
            'image': img_tensor,  # torch.Size([3, h, w]), value [0.0, 1.0]
            'mask': mask_tensor  # torch.Size([h, w]), value [0, 255]
        }
