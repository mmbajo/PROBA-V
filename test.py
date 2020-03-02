import logging
import argparse
import os
import tensorflow as tf
import numpy as np
from skimage import io
from tqdm import tqdm
from models.modelsTF import WDSRConv3D
from utils.utils import *
import imageio.core.util

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logger = logging.getLogger('__name__')


def ignore_warnings(*args, **kwargs):
    pass


imageio.core.util._precision_warn = ignore_warnings


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='/home/mark/DataBank/PROBA-V-CHKPT/patchesDir')
    parser.add_argument('--modelckpt', type=str, default='modelInfo/ckpt_38_top7_90p')
    parser.add_argument('--output', type=str, default='/home/mark/DataBank/PROBA-V-CHKPT/TRAINout_patch38_top7_90p')
    parser.add_argument('--band', type=str, default='RED')
    parser.add_argument('--totest', type=str, default='TEST')
    opt = parser.parse_args()
    return opt


def main():
    logger.info('[ INFO ] Loading data...')
    patchLR = np.load(os.path.join(opt.images, f'{opt.totest}patchesLR_{opt.band}.npy'), allow_pickle=True)
    patchLR = patchLR.transpose((0, 1, 4, 5, 2, 3))

    datasetAllMean = 8818.0603
    datasetAllStd = 6534.1132

    logger.info('[ INFO ] Instantiate model...')
    modelIns = WDSRConv3D(name='patch38', band=opt.band, mean=datasetAllMean, std=datasetAllStd, maxShift=6)
    logger.info('[ INFO ] Building model...')
    model = modelIns.build(scale=3, numFilters=32, kernelSize=(3, 3, 3), numResBlocks=8,
                           expRate=8, decayRate=0.8, numImgLR=7, patchSizeLR=38, isGrayScale=True)

    ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                               psnr=tf.Variable(1.0),
                               model=model)
    ckptDir = os.path.join(opt.modelckpt, opt.band)
    ckptMngr = tf.train.CheckpointManager(checkpoint=ckpt,
                                          directory=ckptDir,
                                          max_to_keep=20)

    ckpt.restore(ckptMngr.latest_checkpoint)
    logger.info('[ INFO ] Generating predictions...')
    y_preds = evaluate(model, patchLR)

    band = opt.band.upper()
    if opt.totest == 'TEST':
        if band == 'NIR':
            i = 1306
        elif band == 'RED':
            i = 1160
    else:
        if band == 'NIR':
            i = 594
        elif band == 'RED':
            i = 0

    if not os.path.exists(opt.output):
        os.makedirs(opt.output)

    logging.info(f'[ SAVE ] Saving predicted images to {opt.output}...')
    for img in tqdm(y_preds):
        io.imsave(os.path.join(opt.output, f"imgset{'%04d' % i}.png"), img[:, :, 0].astype(np.uint16))
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
