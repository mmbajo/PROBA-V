from typing import List

import tensorflow as tf


class Losses:
    '''
    We put all losses and or metrics in one class so that tensorflow detects this loss as one entity...
    '''

    def __init__(self, targetShape=(96, 96, 1), cropBorder=3, bitDepth=16):
        self.targetShapeHeight = targetShape[0]
        self.targetShapeWidth = targetShape[1]
        self.targetShapeChannels = targetShape[2]
        self.cropBorder = cropBorder
        self.maxPixelShift = 2 * cropBorder
        self.numBytes = 2**bitDepth - 1

        self.cropSizeHeight = self.targetShapeHeight - self.maxPixelShift
        self.cropSizeWidth = self.targetShapeWidth - self.maxPixelShift

    def shiftCompensatedcPSNR(self, patchHR: tf.Tensor, maskHR: tf.Tensor, predPatchHR: tf.Tensor) -> tf.Tensor:
        '''
        The maximum cPSNR of every possible pixel shift between the predicted HR image and the ground truth.
        This is how the ESA has been computing the submissions of the contestants.
        See details at the ff link: https://kelvins.esa.int/proba-v-super-resolution/scoring/
        '''
        cropPrediction = cropImage(predPatchHR, self.cropBorder, self.cropSizeHeight,
                                   self.cropBorder, self.cropSizeWidth)
        cachecPSNR = []

        # Iterate through all possible shift configurations
        for i in range(self.maxPixelShift+1):
            for j in range(self.maxPixelShift+1):
                self.stackcPSNR(i, j, patchHR, maskHR, cropPrediction, cachecPSNR)
        cachecPSNR = tf.stack(cachecPSNR)
        cPSNR = tf.reduce_max(cachecPSNR)
        return cPSNR

    def shiftCompensatedL2Loss(self, patchHR: tf.Tensor, maskHR: tf.Tensor, predPatchHR: tf.Tensor) -> tf.Tensor:
        '''
        The minimum L2 Loss of every possible pixel shift between the predicted HR image and the ground truth.
        This is how the ESA has been computing the submissions of the contestants.
        See details at the ff link: https://kelvins.esa.int/proba-v-super-resolution/scoring/
        '''
        cropPrediction = cropImage(predPatchHR, self.cropBorder, self.cropSizeHeight,
                                   self.cropBorder, self.cropSizeWidth)
        cacheLosses = []

        # Iterate through all possible shift configurations
        for i in range(self.maxPixelShift+1):
            for j in range(self.maxPixelShift+1):
                self.stackL2Loss(i, j, patchHR, maskHR, cropPrediction, cacheLosses)
        cacheLosses = tf.stack(cacheLosses)
        minLoss = tf.reduce_min(cacheLosses)
        return minLoss

    def shiftCompensatedL1Loss(self, patchHR: tf.Tensor, maskHR: tf.Tensor, predPatchHR: tf.Tensor) -> tf.Tensor:
        cropPrediction = cropImage(predPatchHR, self.cropBorder, self.cropSizeHeight,
                                   self.cropBorder, self.cropSizeWidth)
        cacheLosses = []

        # Iterate through all possible shift configurations
        for i in range(self.maxPixelShift+1):
            for j in range(self.maxPixelShift+1):
                self.stackL1Loss(i, j, patchHR, maskHR, cropPrediction, cacheLosses)
        cacheLosses = tf.stack(cacheLosses)
        minLoss = tf.reduce_min(cacheLosses)
        return minLoss

    def stackL1Loss(self, i: int, j: int, patchHR: tf.Tensor, maskHR: tf.Tensor, cropPred: tf.Tensor, cache: List[float]):
        cropTrueImg = cropImage(patchHR, i, self.cropSizeHeight, j, self.cropSizeWidth)
        cropTrueMsk = cropImage(maskHR, i, self.cropSizeHeight, j, self.cropSizeWidth)
        cropPredMskd = cropPred * cropTrueMsk
        totalClearPixels = tf.reduce_sum(cropTrueMsk, axis=(1, 2, 3))

        b = self.computeBiasBrightness(totalClearPixels, cropTrueImg, cropPredMskd)

        correctedCropPred = cropPred + b
        correctedCropPredMskd = correctedCropPred * cropTrueMsk

        L1Loss = self.computeL1Loss(totalClearPixels, cropTrueImg, correctedCropPredMskd)
        cache.append(L1Loss)

    def stackL2Loss(self, i: int, j: int, patchHR: tf.Tensor, maskHR: tf.Tensor, cropPred: tf.Tensor, cache: List[float]):
        cropTrueImg = cropImage(patchHR, i, self.cropSizeHeight, j, self.cropSizeWidth)
        cropTrueMsk = cropImage(maskHR, i, self.cropSizeHeight, j, self.cropSizeWidth)
        cropPredMskd = cropPred * cropTrueMsk
        totalClearPixels = tf.reduce_sum(cropTrueMsk, axis=(1, 2, 3))

        b = self.computeBiasBrightness(totalClearPixels, cropTrueImg, cropPredMskd)

        correctedCropPred = cropPred + b
        correctedCropPredMskd = correctedCropPred * cropTrueMsk

        L2Loss = self.computeL2Loss(totalClearPixels, cropTrueImg, correctedCropPredMskd)
        cache.append(L2Loss)

    def stackcPSNR(self, i: int, j: int, patchHR: tf.Tensor, maskHR: tf.Tensor, cropPred: tf.Tensor, cache: List[float]):
        cropTrueImg = cropImage(patchHR, i, self.cropSizeHeight, j, self.cropSizeWidth)
        cropTrueMsk = cropImage(maskHR, i, self.cropSizeHeight, j, self.cropSizeWidth)
        cropPredMskd = cropPred * cropTrueMsk
        totalClearPixels = tf.reduce_sum(cropTrueMsk, axis=(1, 2, 3))

        b = self.computeBiasBrightness(totalClearPixels, cropTrueImg, cropPredMskd)

        correctedCropPred = cropPred + b
        correctedCropPredMskd = correctedCropPred * cropTrueMsk

        cPSNR = self.computecPSNR(totalClearPixels, cropTrueImg, correctedCropPredMskd)
        cache.append(cPSNR)

    def computeBiasBrightness(self, totalClearPixels, HR, SR):
        theShape = tf.shape(HR)
        b = (1.0 / totalClearPixels) * tf.reduce_sum(tf.subtract(HR, SR), axis=(1, 2, 3))
        # Try each bias and find the one with the lowest loss.
        b = tf.reshape(b, (theShape[0], 1, 1, 1))
        return b

    def computeL1Loss(self, totalClearPixels, HR, correctedSR):
        loss = (1.0 / totalClearPixels) * tf.reduce_sum(tf.abs(tf.subtract(HR, correctedSR)), axis=(1, 2, 3))
        return loss

    def computeL2Loss(self, totalClearPixels, HR, correctedSR):
        loss = (1.0 / totalClearPixels) * tf.reduce_sum(tf.square(tf.subtract(HR, correctedSR)), axis=(1, 2, 3))
        return loss

    def computecPSNR(self, totalClearPixels, HR, correctedSR):
        loss = self.computeL2Loss(totalClearPixels, HR, correctedSR)
        cPSNR = 10.0 * (tf.math.log(self.numBytes**2/loss) / tf.math.log(tf.constant(10, dtype=tf.float32)))
        return cPSNR


def cropImage(imgBatch: tf.Tensor, startIdxH: int, lengthHeight: int,
              startIdxW: int, lengthWidth: int) -> tf.Tensor:
    return tf.cast(imgBatch[:, startIdxH: startIdxH + lengthHeight, startIdxW: startIdxW + lengthWidth, :], tf.float32)
