import argparse
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import num_classes
from dataset.cityscapes.labels import trainId2label
from unet import UNet


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=num_classes)

    logging.info('Loading model {}'.format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded !')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')

        img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)  # shape (h, w, 3)

        pred = predict_img(net=net,
                           full_img=img,
                           scale=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        color = cv2.cvtColor(np.asarray(pred['color']), cv2.COLOR_RGB2BGR)
        mask = cv2.addWeighted(img, 0.5, color, 0.5, 0)

        if not args.no_save:
            out_filename = out_files[i]
            path, ext = os.path.splitext(out_filename)
            pred['color'].save(f'{path}_color{ext}')
            pred['classes'].save(f'{path}_classes{ext}')
            cv2.imwrite(f'{path}_mask{ext}', mask)

            logging.info('Mask saved to '
                         f'{path}_color{ext}, '
                         f'{path}_classes{ext}, '
                         f'{path}_mask{ext}')


def predict_img(net, full_img: np.ndarray, device, scale=1, out_threshold=0.5):
    net.eval()

    img = preprocess(full_img, scale)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
    probs = F.softmax(output, dim=1)
    # torch.Size([c, h, w])
    probs = probs.squeeze(0)
    probs_arr = probs.cpu().numpy()

    # torch.Size([c, h, w])
    thres_probs = probs_arr * (probs_arr > out_threshold)

    color_arr = channel2color(thres_probs)
    classes_arr = channel2classes(thres_probs)
    color_arr = restore(color_arr, full_img.shape[:2])
    classes_arr = restore(classes_arr, full_img.shape[:2])
    color_pil = Image.fromarray(color_arr.astype('uint8'))
    classes_pil = Image.fromarray(classes_arr.astype('uint8'))

    return {
        'color': color_pil,
        'classes': classes_pil,
    }


def preprocess(img_arr: np.ndarray, scale: float) -> torch.Tensor:
    '''Process the image to make it can input the network

    Args:
        img_arr (np.ndarray)

    Returns:
        tensor, torch.Size([1, c, h, w])
    '''
    height, width, _ = img_arr.shape
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((round(height * scale), round(width * scale))),
    ])
    img_tensor = tf(img_arr)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def restore(img_arr: np.ndarray, shape):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(shape, interpolation=Image.NEAREST),
    ])
    res_arr = tf(img_arr).squeeze(0).numpy()
    if len(res_arr.shape) > 2:
        res_arr = res_arr.transpose((1, 2, 0))
    return res_arr


def channel2color(probs: np.ndarray):
    probs_trans = probs.transpose((1, 2, 0))
    height, width, _ = probs_trans.shape
    color_result = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            channel_arr = probs_trans[y][x]
            idx = np.argmax(channel_arr)
            if channel_arr[idx] == 0:
                color_result[y][x] = trainId2label[255].color
                continue
            color_result[y][x] = trainId2label[idx].color
    return color_result


def channel2classes(probs: np.ndarray):
    probs_trans = probs.transpose((1, 2, 0))
    height, width, _ = probs_trans.shape
    classes_result = np.zeros((height, width), dtype=np.int64)
    for y in range(height):
        for x in range(width):
            channel_arr = probs_trans[y][x]
            idx = np.argmax(channel_arr)
            if channel_arr[idx] == 0:
                classes_result[y][x] = 0
                continue
            classes_result[y][x] = idx
    return classes_result


def get_args():
    parser = argparse.ArgumentParser(
        description='Predict masks from input images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model',
                        '-m',
                        default='MODEL.pth',
                        metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input',
                        '-i',
                        metavar='INPUT',
                        nargs='+',
                        help='filenames of input images',
                        required=True)

    parser.add_argument('--output',
                        '-o',
                        metavar='INPUT',
                        nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--no-save',
                        '-n',
                        action='store_true',
                        help='Do not save the output masks',
                        default=False)
    parser.add_argument(
        '--mask-threshold',
        '-t',
        type=float,
        help='Minimum probability value to consider a mask pixel white',
        default=0.0)
    parser.add_argument('--scale',
                        '-s',
                        type=float,
                        help='Scale factor for the input images',
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append(f'{pathsplit[0]}_OUT{pathsplit[1]}')
    elif len(in_files) != len(args.output):
        logging.error(
            'Input files and output files are not of the same length')
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


if __name__ == '__main__':
    main()
