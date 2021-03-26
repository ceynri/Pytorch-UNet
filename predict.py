import argparse
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.cityscapes.dataset import CityscapesDataset
from utils.cityscapes.labels import trainId2label


def main():
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=19)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = cv2.imread(fn, flags=cv2.IMREAD_COLOR)  # shape (h, w, 3)

        mask = predict_img(net=net,
                           full_img=img,
                           scale=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            mask.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)


def predict_img(net,
                full_img: np.ndarray,
                device,
                scale=1,
                out_threshold=0.5):
    net.eval()

    img = preprocess(full_img, scale)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
    probs = F.softmax(output, dim=1)
    probs = probs.squeeze(0)  # torch.Size([c, h, w])
    probs_arr = probs.cpu().numpy()

    # tf = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize(full_img.size[1]),
    # ])

    thres_probs = probs_arr * (probs_arr > out_threshold)
    color_arr = channel2color(thres_probs)
    result = Image.fromarray(color_arr.astype('uint8'))
    return result


def preprocess(img_arr: np.ndarray, scale) -> torch.Tensor:
    '''Process the image to make it can input the network

    Args:
        img_arr (np.ndarray)

    Returns:
        tensor, torch.Size([1, c, h, w])
    '''
    height, width, _  = img_arr.shape
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((int(height * scale), int(width * scale))),
    ])
    img_tensor = tf(img_arr)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def channel2color(probs:np.ndarray):
    probs_trans = probs.transpose((1, 2, 0))
    height, width, n_channels = probs_trans.shape
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

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


if __name__ == "__main__":
    main()
