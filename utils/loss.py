from typing import List

import tensorflow as tf


# Define Loss constants
CROP_BORDER = 3
MAX_PIXEL_SHIFT = 2*CROP_BORDER


def shiftCompensatedcPSNR(patchHR: tf.Tensor, maskHR: tf.Tensor, predPatchHR: tf.Tensor) -> float:
    '''
    The maximum cPSNR of every possible pixel shift between the predicted HR image and the ground truth.
    This is how the ESA has been computing the submissions of the contestants.
    See details at the ff link: https://kelvins.esa.int/proba-v-super-resolution/scoring/
    '''
    N, H, W, C = tf.shape(patchHR)

    cropSizeHeight = H - MAX_PIXEL_SHIFT
    cropSizeWidth = W - MAX_PIXEL_SHIFT
    cropPrediction = cropImage(predPatchHR, CROP_BORDER, cropSizeHeight, CROP_BORDER, cropSizeWidth)
    cachecPSNR = []

    # Iterate through all possible shift configurations
    for i in range(MAX_PIXEL_SHIFT+1):
        for j in range(MAX_PIXEL_SHIFT+1):
            stackcPSNR(i, j, patchHR, maskHR, cropPrediction, cachecPSNR)
    cachecPSNR = tf.stack(cachecPSNR)
    maxcPSNR = tf.reduce_max(cachecPSNR)
    return maxcPSNR


def shiftCompensatedL2Loss(patchHR: tf.Tensor, maskHR: tf.Tensor, predPatchHR: tf.Tensor) -> float:
    '''
    The minimum L2 Loss of every possible pixel shift between the predicted HR image and the ground truth.
    This is how the ESA has been computing the submissions of the contestants.
    See details at the ff link: https://kelvins.esa.int/proba-v-super-resolution/scoring/
    '''
    N, H, W, C = tf.shape(patchHR)

    cropSizeHeight = H - MAX_PIXEL_SHIFT
    cropSizeWidth = W - MAX_PIXEL_SHIFT
    cropPrediction = cropImage(predPatchHR, CROP_BORDER, cropSizeHeight, CROP_BORDER, cropSizeWidth)
    cacheLosses = []

    # Iterate through all possible shift configurations
    for i in range(MAX_PIXEL_SHIFT+1):
        for j in range(MAX_PIXEL_SHIFT+1):
            stackL2Loss(i, j, patchHR, maskHR, cropPrediction, cacheLosses)
    cacheLosses = tf.stack(cacheLosses)
    minLoss = tf.reduce_min(cacheLosses)
    return minLoss


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

    # Iterate through all possible shift configurations
    for i in range(MAX_PIXEL_SHIFT+1):
        for j in range(MAX_PIXEL_SHIFT+1):
            stackL1Loss(i, j, patchHR, maskHR, cropPrediction, cacheLosses)
    cacheLosses = tf.stack(cacheLosses)
    minLoss = tf.reduce_min(cacheLosses)
    return minLoss


def stackL1Loss(i: int, j: int, patchHR: tf.Tensor, maskHR: tf.Tensor, cropPred: tf.Tensor, cache: List[float]):
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


def stackL2Loss(i: int, j: int, patchHR: tf.Tensor, maskHR: tf.Tensor, cropPred: tf.Tensor, cache: List[float]):
    N, cropSizeHeight, cropSizeWidth, C = tf.shape(cropPred)
    cropTrueImg = cropImage(patchHR, i, cropSizeHeight, j, cropSizeWidth)
    cropTrueMsk = cropImage(maskHR, i, cropSizeHeight, j, cropSizeWidth)
    cropPredMskd = cropPred * cropTrueMsk
    totalClearPixels = tf.reduce_sum(cropTrueMsk, axis=(1, 2, 3))

    b = computeBiasBrightness(totalClearPixels, cropTrueImg, cropPredMskd)

    correctedCropPred = cropPred + b
    correctedCropPredMskd = correctedCropPred * cropTrueMsk

    L2Loss = computeL2Loss(totalClearPixels, cropTrueImg, correctedCropPredMskd)
    cache.append(L2Loss)


def stackcPSNR(i: int, j: int, patchHR: tf.Tensor, maskHR: tf.Tensor, cropPred: tf.Tensor, cache: List[float]):
    N, cropSizeHeight, cropSizeWidth, C = tf.shape(cropPred)
    cropTrueImg = cropImage(patchHR, i, cropSizeHeight, j, cropSizeWidth)
    cropTrueMsk = cropImage(maskHR, i, cropSizeHeight, j, cropSizeWidth)
    cropPredMskd = cropPred * cropTrueMsk
    totalClearPixels = tf.reduce_sum(cropTrueMsk, axis=(1, 2, 3))

    b = computeBiasBrightness(totalClearPixels, cropTrueImg, cropPredMskd)

    correctedCropPred = cropPred + b
    correctedCropPredMskd = correctedCropPred * cropTrueMsk

    cPSNR = computecPSNR(totalClearPixels, cropTrueImg, correctedCropPredMskd)
    cache.append(cPSNR)


def computeL1Loss(totalClearPixels, HR, correctedSR):
    loss = (1.0 / totalClearPixels) * tf.reduce_sum(tf.abs(tf.subtract(HR, correctedSR)), axis=(1, 2))
    return loss


def computeL2Loss(totalClearPixels, HR, correctedSR):
    loss = (1.0 / totalClearPixels) * tf.reduce_sum(tf.square(tf.subtract(HR, correctedSR)), axis=(1, 2))
    return loss


def computecPSNR(totalClearPixels, HR, correctedSR):
    loss = (1.0 / totalClearPixels) * tf.reduce_sum(tf.square(tf.subtract(HR, correctedSR)), axis=(1, 2))
    cPSNR = -tf.math.log(loss) / tf.math.log(tf.constant(10, dtype=tf.float32))
    return cPSNR


def computeBiasBrightness(totalClearPixels, HR, SR):
    N, H, W, C = tf.shape(HR)
    b = (1.0 / totalClearPixels) * tf.reduce_sum(tf.subtract(HR, SR), axis=(1, 2))
    b = tf.reshape(b, (N, 1, 1, C))
    return b


def cropImage(imgBatch: tf.Tensor, startIdxH: int, lengthHeight: int,
              startIdxW: int, lengthWidth: int) -> tf.Tensor:
    return tf.cast(imgBatch[:, startIdxH: startIdxH + lengthHeight, startIdxW: startIdxW + lengthWidth, :], tf.float32)
