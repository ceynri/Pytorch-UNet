import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.cityscapes.dataset import CityscapesDataset
from torch.utils.data import DataLoader

# dir_cityscapes = '/PATH_TO_CITYSCAPES/'
dir_cityscapes = '/data/chenyangrui/cityscapes/'
dir_imgs = os.path.join(dir_cityscapes, 'leftImg8bit_trainvaltest/leftImg8bit/')
dir_masks = os.path.join(dir_cityscapes, 'gtFine_trainvaltest/gtFine/')
dir_imgs_train = os.path.join(dir_imgs, 'train')
dir_masks_train = os.path.join(dir_masks, 'train')
dir_imgs_val = os.path.join(dir_imgs, 'val')
dir_masks_val = os.path.join(dir_masks, 'val')
dir_checkpoint = '/data/chenyangrui/unet/checkpoints/'


def main():
    # prepare
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device:
        device = torch.device(args.device)
    logging.info(f'Using device {device}')

    # get net
    net = UNet(n_channels=3, n_classes=19, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    # load model
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    # train
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  continue_epoch=args.continue_epoch,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  cp_name=args.checkpointname)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), './tmp/INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


def train_net(net,
              device,
              epochs=5,
              continue_epoch=0,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              img_scale=1,
              cp_name='CP'):
    # load dataset
    train_dataset = CityscapesDataset(dir_imgs_train, dir_masks_train, scale=img_scale)
    val_dataset = CityscapesDataset(dir_imgs_val, dir_masks_val, scale=img_scale)
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # info log
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    for epoch in range(continue_epoch, epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']  # torch.Size([b, c, h, w])
                true_masks = batch['mask']  # torch.Size([b, h, w])
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                masks_pred = net(imgs)  # torch.Size([b, c, h, w])
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 5
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'{cp_name}_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-c', '--continue-epoch', type=int, default=0,
                        help='Continue training starting with the epoch', dest='continue_epoch')
    parser.add_argument('-b', '--batch-size', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-d', '--device', dest='device', type=str, default='',
                        help='Appoint device to train')
    parser.add_argument('-n', '--checkpoint-name', dest='checkpointname', type=str, default='CP',
                        help='set saved checkpoint name')

    return parser.parse_args()


if __name__ == '__main__':
    main()
