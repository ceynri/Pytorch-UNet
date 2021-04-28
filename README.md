# UNet: semantic segmentation with PyTorch

This project is forked from <https://github.com/milesial/Pytorch-UNet> and made some adjustments and improvements to adapt to the cityscape dataset.

Provide only a minimal introduction and support.

## Usage

**Note : Use Python 3.6 or newer**

### Prediction

After training your model and saving it to MODEL.pth, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

```shell script
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] [--output INPUT [INPUT ...]] [--no-save] [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white (default: 0.0)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 0.5)
```

You can specify which model file to use with `--model MODEL.pth`.

### Training

```shell script
> python train.py -h
usage: train.py [-h] [-e EPOCHS] [-c CONTINUE_EPOCH] [-b [BATCHSIZE]] [-l [LR]] [-f LOAD] [-s SCALE] [-d DEVICE] [-n CHECKPOINTNAME] [-a]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs (default: 20)
  -c CONTINUE_EPOCH, --continue-epoch CONTINUE_EPOCH
                        Continue training starting with the epoch (default: 0)
  -b [BATCHSIZE], --batch-size [BATCHSIZE]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.0001)
  -f LOAD, --load LOAD  Load model from a .pth file (default: None)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -d DEVICE, --device DEVICE
                        Appoint device to train (default: )
  -n CHECKPOINTNAME, --checkpoint-name CHECKPOINTNAME
                        set saved checkpoint name (default: CP)
  -a, --add-extra       add coarse train dataset (default: False)
```

By default, the `scale` is 1, so if you wish to obtain better results (but use more memory), set it to 1.

The input images and target masks path set by config.py.

## Tensorboard

You can visualize in real time the train and test losses, the weights and gradients, along with the model predictions with tensorboard:

`tensorboard --logdir=runs`

You can find a reference training run with the Caravana dataset on [TensorBoard.dev](https://tensorboard.dev/experiment/1m1Ql50MSJixCbG1m9EcDQ/#scalars&_smoothingWeight=0.6) (only scalars are shown currently).

## Training condition

The model has be trained from scratch on a TITAN X 12GB.

## Others

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
