import os

# dir_cityscapes = '/PATH_TO_CITYSCAPES/'
dir_cityscapes = '/data/chenyangrui/cityscapes/'

dir_imgs = os.path.join(dir_cityscapes, 'leftImg8bit_trainvaltest/leftImg8bit/')
dir_masks = os.path.join(dir_cityscapes, 'gtFine_trainvaltest/gtFine/')
dir_extra_imgs = os.path.join(dir_cityscapes, 'leftImg8bit_trainextra/leftImg8bit/')
dir_extra_masks = os.path.join(dir_cityscapes, 'gtCoarse/gtCoarse/')

dir_imgs_train = os.path.join(dir_imgs, 'train')
dir_masks_train = os.path.join(dir_masks, 'train')
dir_extra_imgs_train = os.path.join(dir_extra_imgs, 'train_extra')
dir_extra_masks_train = os.path.join(dir_extra_masks, 'train_extra')

dir_imgs_val = os.path.join(dir_imgs, 'val')
dir_masks_val = os.path.join(dir_masks, 'val')

dir_checkpoint = '/data/chenyangrui/unet/checkpoints/'

num_classes = 19
