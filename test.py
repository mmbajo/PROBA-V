import logging
import argparse
import os
import tensorflow as tf
import numpy as np
from skimage import io
from tqdm import tqdm
from models.modelsTF import WDSRConv3D
from utils.parseConfig import parseConfig
from utils.utils import *
import imageio.core.util

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logger = logging.getLogger('__name__')


def ignore_warnings(*args, **kwargs):
    pass


imageio.core.util._precision_warn = ignore_warnings


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfg/FINAL.cfg', type=str)
    parser.add_argument('--band', type=str, default='RED')
    parser.add_argument('--totest', type=str, default='TEST')
    opt = parser.parse_args()
    return opt


def main(config):
    logger.info('[ INFO ] Loading data...')
    dataDir = os.path.join(config['preprocessing_out'], 'resolverDir')
    patchLR = np.load(os.path.join(dataDir, f'{opt.totest}patchesLR_{opt.band}.npy'), allow_pickle=True)
    patchLR = patchLR.transpose((0, 1, 4, 5, 2, 3))

    if opt.band == 'NIR':
        datasetAllMean = 8075.2045  # 8818.0603
        datasetAllStd = 3160.7272  # 6534.1132
    else:
        datasetAllMean = 5266.2245
        datasetAllStd = 3431.8614

    logger.info('[ INFO ] Instantiate model...')
    modelIns = WDSRConv3D(name='superResolutionNet', band=opt.band,
                          mean=datasetAllMean, std=datasetAllStd, maxShift=config['max_shift'])

    logger.info('[ INFO ] Building model...')
    kernelSize = (config['kernel_size'], config['kernel_size'], config['kernel_size'])
    model = modelIns.build(scale=config['scale'], numFilters=config['num_filters'], kernelSize=kernelSize,
                           numResBlocks=config['num_res_blocks'], expRate=config['exp_rate'],
                           decayRate=config['decay_rate'], numImgLR=config['num_low_res_imgs'],
                           patchSizeLR=config['patch_size'], isGrayScale=config['is_grayscale'])

    ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                               psnr=tf.Variable(1.0),
                               model=model)
    basename = os.path.basename(opt.cfg).split('.')[0]
    ckptDir = os.path.join(config['model_out'], f'ckpt_{basename}', opt.band)
    ckptMngr = tf.train.CheckpointManager(checkpoint=ckpt,
                                          directory=ckptDir,
                                          max_to_keep=5)

    ckpt.restore(ckptMngr.latest_checkpoint)
    logger.info('[ INFO ] Generating predictions...')
    y_preds = evaluate(model, patchLR)

    band = opt.band.upper()
    if not os.path.exists(f'removedTrainSets{band}.txt'):
        toOmit = []
    else:
        with open(f'removedTrainSets{band}.txt', 'r') as f:
            toOmit = f.readlines()
        toOmit = [int(float(x.split('\n')[0])) for x in toOmit]

    if opt.totest == 'TEST':
        outDir = config['test_out'] + f'_{basename}'
        if band == 'NIR':
            i = 1306
        elif band == 'RED':
            i = 1160
    else:
        outDir = config['train_out'] + f'_{basename}'
        if band == 'NIR':
            i = 594
        elif band == 'RED':
            i = 0

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    logging.info(f'[ SAVE ] Saving predicted images to {outDir}...')
    for img in tqdm(y_preds):
        while i in toOmit:
            i += 1
        io.imsave(os.path.join(outDir, f"imgset{'%04d' % i}.png"), img[:, :, 0].astype(np.uint16))
        i += 1


def evaluate(model, X_test_patches):
    y_preds = []

    for i in tqdm(range(0, X_test_patches.shape[0])):
        # Resolve
        res_patches = resolveByBatch(model, X_test_patches[i])
        y_pred = reconstruct_from_patches(res_patches)
        y_preds.append(y_pred)
    return y_preds


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)

    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 2**16)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.float32)
    sr_batch = np.array(sr_batch)
    return sr_batch


def resolveByBatch(model, lr_batch, batch_size=16):
    n, rem = divmod(lr_batch.shape[0], batch_size)
    cache = []
    for i in range(1, n+1):
        sr_batch = resolve(model, lr_batch[batch_size*(i-1): batch_size*i])
        cache.append(sr_batch)
    if rem:
        sr_batch = resolve(model, lr_batch[batch_size*n: batch_size*n + rem])
        cache.append(sr_batch)
    return np.concatenate(cache)


def resolveBySampleAveraging(model, lr_batch):
    cache = []
    for _ in range(20):
        newIdx = np.random.permutation(lr_batch.shape[3])
        lr_batch = lr_batch[:, :, :, newIdx, :]
        resPatches = resolve(model, lr_batch)
        cache.append(resPatches)
    toAve = tf.stack(cache)
    sr_batch = tf.reduce_mean(toAve, axis=0)
    return sr_batch


def reconstruct_from_patches(images):
    rec_img = np.zeros((384, 384, 1))
    block_n = 0
    n = int(len(images) ** 0.5)
    patchSize = images.shape[1]
    for i in range(1, n+1):
        for j in range(1, n+1):

            rec_img[(i-1)*patchSize: i*patchSize, (j-1)*patchSize: j*patchSize] = images[block_n, :, :, ]
            block_n += 1

    return rec_img.reshape((384, 384, 1))


if __name__ == '__main__':
    opt = parser()
    config = parseConfig(opt.cfg)
    main(config)
