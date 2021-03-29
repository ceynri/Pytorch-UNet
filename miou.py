import argparse
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import *
from unet import UNet
from utils.cityscapes.dataset import CityscapesDataset


def fast_hist(pred: np.ndarray, gt: np.ndarray, n: int):
    '''
    Args:
        gt: true label array
        pred: pred label array
        n: num_classes
	'''
    gt = gt.flatten()
    pred = pred.flatten()
    # filter array (boolean)
    k = (gt >= 0) & (gt < n)
    # np.bincount  calculated the number of occurrences of each of the n**2 numbers
    # from 0 to n**2-1, and the return value shape (n, n)
    return np.bincount(n * gt[k].astype(int) + pred[k],
                       minlength=n**2).reshape(n, n)


def per_class_iou(hist):
    '''
    Calculate IoU for each classes, the shape of hist (n, n)
    The one-dimensional array composed of the values on the diagonal of the matrixis 
    divided by the sum of all the elements of the matrix, and the return value shape is (n,)
	'''
    np.seterr(divide='ignore', invalid='ignore')
    iou_arr = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    np.seterr(divide='warn', invalid='warn')
    return iou_arr


def calc_miou(pred: np.ndarray, gt: np.ndarray, n_classes: int):
    hist = fast_hist(pred, gt, n_classes)
    iou_arr = per_class_iou(hist)
    return np.nanmean(iou_arr)


def batch_calc(args):
    from eval import eval_net

    # get net
    net = UNet(n_channels=3, n_classes=num_classes)

    logging.info(f'Loading model {args.model}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info('Model loaded !')

    val_dataset = CityscapesDataset(dir_imgs_val,
                                    dir_masks_val,
                                    scale=args.scale)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True)
    miou = eval_net(net, val_loader, device, type='miou')
    logging.info(f'total mIoU value: {miou}')
    return miou


def single_calc(args):
    # get img array
    gt = cv2.imread(args.gt, flags=cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(args.pred, flags=cv2.IMREAD_GRAYSCALE)
    miou = calc_miou(pred, gt, num_classes)
    logging.info(f'mIoU value: {miou}')
    return miou


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = get_args()
    if args.single:
        single_calc(args)
    else:
        batch_calc(args)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        '--single',
                        type=bool,
                        default=False,
                        help='Whether to calculate only one image',
                        dest='single')
    # 存放验证集分割标签的文件夹
    parser.add_argument('-g',
                        '--gt',
                        dest="gt",
                        type=str,
                        help='ground true image\'s directory or file path')
    # 存放验证集分割结果的文件夹
    parser.add_argument('-p',
                        '--pred',
                        dest="pred",
                        type=str,
                        help='pred images\'s directory or file path')

    '''以下是批处理的参数'''

    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        nargs='?',
                        default=1,
                        help='Batch size',
                        dest='batchsize')
    parser.add_argument('-m',
                        '--model',
                        dest='model',
                        type=str,
                        help='Load model from a .pth file')
    parser.add_argument('-s',
                        '--scale',
                        dest='scale',
                        type=float,
                        default=1,
                        help='Downscaling factor of the images')
    return parser.parse_args()


if __name__ == "__main__":
    main()
