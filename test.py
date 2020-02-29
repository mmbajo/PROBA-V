import argparse
import os
import tensorflow as tf
import numpy as np
from skimage import io
from tqdm import tqdm
from modelsTF import *

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logger = logging.getLogger('__name__')


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='/home/mark/DataBank/PROBA-V-CHKPT/patchesDir')
    parser.add_argument('--modelckpt', type=str, default='/home/mark/DataBank/ckptNewRed')
    parser.add_argument('--output', type=str, default='/home/mark/DataBank/PROBA-V-CHKPT/testout')
    parser.add_argument('--band', type=str, default='RED')
    opt = parser.parse_args()
    return opt


def main():
    X = np.load(os.path.join(opt.images, f'TESTpatchesLR_{opt.band}.npy'), allow_pickle=True)
    X = X.transpose((0, 1, 4, 5, 2, 3))
    model = WDSRConv3D(scale=3, numFilters=32, kernelSize=(3, 3, 3), numResBlocks=8,
                       expRate=8, decayRate=0.8, numImgLR=9, patchSizeLR=32, isGrayScale=True)

    ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                               psnr=tf.Variable(1.0),
                               model=model)

    ckptMngr = tf.train.CheckpointManager(checkpoint=ckpt,
                                          directory=opt.modelckpt,
                                          max_to_keep=5)

    ckpt.restore(ckptMngr.latest_checkpoint)
    y_preds = evaluate(model, X)

    band = opt.band.upper()
    if band == 'NIR':
        i = 1306
    elif band == 'RED':
        i = 1160

    if not os.path.exists(opt.output):
        os.makedirs(opt.output)

    logging.info(f'Saving predicted images to {opt.output}')
    for img in tqdm(y_preds):
        io.imsave(os.path.join(opt.output, f'imgset{i}.png'), img[:, :, 0].astype(np.uint16))
        i += 1


def evaluate(model, X_test_patches):
    y_preds = []

    for i in tqdm(range(0, X_test_patches.shape[0])):
        # Resolve
        res_patches = resolve(model, X_test_patches[i])
        y_pred = reconstruct_from_patches(np.array(res_patches))
        y_preds.append(y_pred)
    return y_preds


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)

    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 2**16)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.float32)
    return sr_batch


def reconstruct_from_patches(images):
    rec_img = np.zeros((384, 384, 1))
    block_n = 0
    first_block = images[0, :, :, ]
    for i in range(1, 5):
        for j in range(1, 5):

            rec_img[(i-1)*96:i*96, (j-1)*96:j*96] = images[block_n, :, :, ]
            block_n += 1

    return rec_img.reshape((384, 384, 1))


if __name__ == '__main__':
    opt = parser()
    main()
