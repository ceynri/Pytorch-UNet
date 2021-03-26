import logging
from glob import glob
from os import listdir, path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# from utils.cityscapes.labels import color2id


class CityscapesDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, img_suffix='_leftImg8bit', mask_suffix='_gtFine_labelTrainIds'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.data = []
        for city in listdir(imgs_dir):
            city_dir = path.join(imgs_dir, city)
            for file in listdir(city_dir):
                if file.startswith('.'):
                    continue
                filename = file.rsplit(sep='_', maxsplit=1)[0]
                img_file = path.join(self.imgs_dir, city, f'{filename}{self.img_suffix}.png')
                mask_file = path.join(self.masks_dir, city, f'{filename}{self.mask_suffix}.png')
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

        # resize & to tensor
        new_size = tuple(int(size * self.scale) for size in img.shape[0:2])
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
