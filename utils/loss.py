from typing import List, Tuple, Dict

import tensorflow as tf
import numpy as np

# Define Loss constants
CROP_BORDER = 3
MAX_PIXEL_SHIFT = 2*CROP_BORDER


def shiftCompensatedL1Loss(patchHR: tf.Tensor, maskHR: tf.Tensor, predPatchHR: tf.Tensor) -> float:
    '''
    The minimum L1 Loss of every possible pixel shift between the predicted HR image and the ground truth.
    This is how the ESA has been computing the submissions of the contestants.
    See details at the ff link: https://kelvins.esa.int/proba-v-super-resolution/scoring/
    '''
    N, H, W, C = tf.shape(patchHR)

    cropSizeHeight = H - MAX_PIXEL_SHIFT
    cropSizeWidth = W - MAX_PIXEL_SHIFT
    cropPrediction = cropImage(predPatchHR, CROP_BORDER, cropSizeHeight, CROP_BORDER, cropSizeWidth)
    cacheLosses = []
    for i in range(MAX_PIXEL_SHIFT+1):
        for j in range(MAX_PIXEL_SHIFT+1):
            stackL1Loss(i, j, patchHR, maskHR, cropPrediction, cacheLosses)
    cacheLosses = tf.stack(cacheLosses)
    minLoss = tf.reduce_min(cacheLosses)
    return minLoss


def stackL1Loss(i: int, j: int, patchHR: tf.Tensor, maskHR: tf.Tensor, cropPred: tf.Tensor, cache: List[tf.float32]):
    N, cropSizeHeight, cropSizeWidth, C = tf.shape(cropPred)
    cropTrueImg = cropImage(patchHR, i, cropSizeHeight, j, cropSizeWidth)
    cropTrueMsk = cropImage(maskHR, i, cropSizeHeight, j, cropSizeWidth)
    cropPredMskd = cropPred * cropTrueMsk
    totalClearPixels = tf.reduce_sum(cropTrueMsk, axis=(1, 2, 3))

    b = computeBiasBrightness(totalClearPixels, cropTrueImg, cropPredMskd)

    correctedCropPred = cropPred + b
    correctedCropPredMskd = correctedCropPred * cropTrueMsk

    L1Loss = computeL1Loss(totalClearPixels, cropTrueImg, correctedCropPredMskd)
    cache.append(L1Loss)


def computeL1Loss(totalClearPixels, HR, correctedSR):
    loss = (1.0 / totalClearPixels) * tf.reduce_sum(tf.abs(tf.subtract(HR, correctedSR)), axis=(1, 2))
    return loss


def computeBiasBrightness(totalClearPixels, HR, SR):
    N, H, W, C = tf.shape(HR)
    b = (1.0 / totalClearPixels) * tf.reduce_sum(tf.subtract(HR, SR), axis=(1, 2))
    b = tf.reshape(b, (N, 1, 1, C))
    return b


def cropImage(imgBatch: tf.tensor, startIdxH: int, lengthHeight: int,
              startIdxW: int, lengthWidth: int) -> tf.tensor:
    return tf.cast(imgBatch[:, startIdxH: startIdxH + lengthHeight, startIdxW: startIdxW + lengthWidth, :], tf.float32)


def PSNR(patchHR: np.ma.masked_array):
    pass
