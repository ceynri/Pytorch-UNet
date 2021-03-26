from os import listdir, path
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
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


    def resize(self, pil_img):
        w, h = pil_img.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH), Image.NEAREST)
        return pil_img


    # def mask2label(self, mask):
    #     new_mask = np.zeros((19,) + mask.shape[0:2])

    #     for y in range(mask.shape[0]):
    #         for x in range(mask.shape[1]):
    #             color = mask[y][x]
    #             if tuple(color) not in color2id:
    #                 continue
    #             new_mask[color2id[color]][y][x] = 1
    #     return new_mask



    # def mask2label(self, mask):
    #     height = mask.shape[0]
    #     width = mask.shape[1]
    #     new_mask = np.zeros((height, width))

    #     for y in range(height):
    #         for x in range(width):
    #             color = mask[y][x]
    #             if tuple(color) not in color2id:
    #                 continue
    #             new_mask[y][x] = color2id[color]
    #     return new_mask


    def transpose(self, img):
        img_trans = img.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans


    def __getitem__(self, i):
        img_file, mask_file = self.data[i]

        img = Image.open(img_file)
        mask = Image.open(mask_file)

        img = self.resize(img)
        mask = self.resize(mask)

        img_arr = np.array(img)
        mask_arr = np.array(mask)

        img_arr = self.transpose(img_arr)

        img_tensor = torch.from_numpy(img_arr).type(torch.FloatTensor)
        mask_tensor = torch.from_numpy(mask_arr).type(torch.FloatTensor)
        return {
            'image': img_tensor,
            'mask': mask_tensor
        }
