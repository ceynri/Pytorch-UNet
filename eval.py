import logging
from predict import channel2classes

import numpy as np
import torch
import torch.nn.functional as F
from config import num_classes
from miou import fast_hist, per_class_iou
from tqdm import tqdm


def eval_net(net, loader, device, type='cross_entropy', n_classes: int = num_classes):
    """
    Evaluation without the densecrf with the dice coefficient.
    Or evaluation with mIoU.
    """
    net.eval()
    n_val = len(loader)  # the number of batch
    total = 0
    hist = np.zeros((n_classes, n_classes))

    with tqdm(total=n_val, desc='Validation round', unit='batch',
              leave=False) as pbar:
        for batch in loader:
            imgs, mask_true = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            with torch.no_grad():
                pred = net(imgs)

            if type == 'cross_entropy':
                total += F.cross_entropy(pred,
                                         mask_true,
                                         ignore_index=255).item()
            elif type == 'miou':
                probs = F.softmax(pred, dim=1)
                probs = probs.squeeze(0)
                probs_arr = probs.cpu().numpy()
                mask_pred = channel2classes(probs_arr)
                mask_true = mask_true.cpu().numpy()
                hist += fast_hist(mask_pred, mask_true, n_classes)
            else:
                logging.error('error eval type!')

            pbar.update()

    net.train()

    if type == 'cross_entropy':
        return total / n_val
    elif type == 'miou':
        mIoUs = per_class_iou(hist)
        return np.nanmean(mIoUs)
