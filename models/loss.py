from typing import List

import tensorflow as tf
from utils.utils import cropImage


# FIX ME: Too many repeated lines
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

        self.pi = 0.7  # This constant is for the SobelL1Mix loss function

        self.cropSizeHeight = self.targetShapeHeight - self.maxPixelShift
        self.cropSizeWidth = self.targetShapeWidth - self.maxPixelShift

        # Initialize gaussian mask for SSIM computation
        # https://en.wikipedia.org/wiki/Structural_similarity
        self.sigma = [0.5, 1., 2., 4., 8.]  # gaussian mask std. dev
        self.C1 = (0.01*self.numBytes)**2
        self.C2 = (0.03*self.numBytes)**2
        self.C3 = self.C2/2
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 1.0
        self.eta = 0.25  # mixture percentage for mL1 loss and SSIM

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
        cPSNR = tf.reduce_max(cachecPSNR, axis=0)
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
        cacheLosses = tf.stack(cacheLosses)  # [49, numBatchSize]
        minLoss = tf.reduce_min(cacheLosses, axis=0)
        return tf.reduce_mean(minLoss)

    def shiftCompensatedL1Loss(self, patchHR: tf.Tensor, maskHR: tf.Tensor, predPatchHR: tf.Tensor) -> tf.Tensor:
        cropPrediction = cropImage(predPatchHR, self.cropBorder, self.cropSizeHeight,
                                   self.cropBorder, self.cropSizeWidth)
        cacheLosses = []

        # Iterate through all possible shift configurations
        for i in range(self.maxPixelShift+1):
            for j in range(self.maxPixelShift+1):
                self.stackL1Loss(i, j, patchHR, maskHR, cropPrediction, cacheLosses)
        cacheLosses = tf.stack(cacheLosses)
        minLoss = tf.reduce_min(cacheLosses, axis=0)
        return tf.reduce_mean(minLoss)

    def shiftCompensatedL1EdgeLoss(self, patchHR: tf.Tensor, maskHR: tf.Tensor, predPatchHR: tf.Tensor) -> tf.Tensor:
        cropPrediction = cropImage(predPatchHR, self.cropBorder, self.cropSizeHeight,
                                   self.cropBorder, self.cropSizeWidth)
        cacheLosses = []

        # Iterate through all possible shift configurations
        for i in range(self.maxPixelShift+1):
            for j in range(self.maxPixelShift+1):
                self.stackL1EdgeLoss(i, j, patchHR, maskHR, cropPrediction, cacheLosses)
        cacheLosses = tf.stack(cacheLosses)
        minLoss = tf.reduce_min(cacheLosses, axis=0)
        return tf.reduce_mean(minLoss)

    def shiftCompensatedRevSSIM(self, patchHR: tf.Tensor, maskHR: tf.Tensor, predPatchHR: tf.Tensor) -> tf.Tensor:
        cropPrediction = cropImage(predPatchHR, self.cropBorder, self.cropSizeHeight,
                                   self.cropBorder, self.cropSizeWidth)
        cacheRevSSIM = []

        # Iterate through all possible shift configurations
        for i in range(self.maxPixelShift+1):
            for j in range(self.maxPixelShift+1):
                self.stackRevSSIM(i, j, patchHR, maskHR, cropPrediction, cacheRevSSIM)
        cacheLosses = tf.stack(cacheRevSSIM)
        minRevSSIM = tf.reduce_min(cacheRevSSIM, axis=0)
        return minRevSSIM

    def stackRevSSIM(self, i: int, j: int, patchHR: tf.Tensor, maskHR: tf.Tensor, cropPred: tf.Tensor, cache: List[float]):
        cropTrueImg = cropImage(patchHR, i, self.cropSizeHeight, j, self.cropSizeWidth)
        cropTrueMsk = cropImage(maskHR, i, self.cropSizeHeight, j, self.cropSizeWidth)
        cropPredMskd = cropPred * cropTrueMsk
        totalClearPixels = tf.reduce_sum(cropTrueMsk, axis=(1, 2, 3))

        b = self.computeBiasBrightness(totalClearPixels, cropTrueImg, cropPredMskd)

        correctedCropPred = cropPred + b
        correctedCropPredMskd = correctedCropPred * cropTrueMsk

        revSSIM = self.computeRevMultiScaleSSIM(maskHR, totalClearPixels, cropTrueImg, correctedCropPredMskd)
        cache.append(revSSIM)

    def stackL1EdgeLoss(self, i: int, j: int, patchHR: tf.Tensor, maskHR: tf.Tensor, cropPred: tf.Tensor, cache: List[float]):
        cropTrueImg = cropImage(patchHR, i, self.cropSizeHeight, j, self.cropSizeWidth)
        cropTrueMsk = cropImage(maskHR, i, self.cropSizeHeight, j, self.cropSizeWidth)
        cropPredMskd = cropPred * cropTrueMsk
        totalClearPixels = tf.reduce_sum(cropTrueMsk, axis=(1, 2, 3))

        b = self.computeBiasBrightness(totalClearPixels, cropTrueImg, cropPredMskd)

        correctedCropPred = cropPred + b
        correctedCropPredMskd = correctedCropPred * cropTrueMsk

        L1Loss = self.computeL1EdgeLoss(totalClearPixels, cropTrueImg, correctedCropPredMskd)
        cache.append(L1Loss)

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

    def computeRevMultiScaleSSIM(self, mask, totalClearPixels, HR, correctedSR, isMixed=True):
        weights = []
        for i in range(len(self.sigma)):
            w = tf.math.exp(-1.0*np.arange(-HR.shape[1]//2, HR.shape[1]//2)/(2*self.sigma[i]**2))
            w = tf.einsum('i,j->ij', w, w) * mask  # masked outer product
            w = w/tf.reduce_sum(w)  # normalize
            w = tf.reshape(w, [1, HR.shape[1], HR.shape[2], HR.shape[3]])
            w = tf.tile(w, [HR.shape[0], 1, 1, HR.shape[3]])
            weights.append(w)
        weights = tf.stack(weights)

        muHR = tf.reduce_sum(weights * HR, axis=(2, 3), keepdims=True)
        muSR = tf.reduce_sum(weights * correctedSR, axis=(2, 3), keepdims=True)
        sigmaHR = tf.reduce_sum(weights * HR**2, axis=(2, 3), keepdims=True) - muHR**2
        sigmaSR = tf.reduce_sum(weights * correctedSR**2, axis=(2, 3), keepdims=True) - muSR**2
        cov = tf.reduce_sum(weights * HR * correctedSR, axis=(2, 3), keepdims=True) - muSR*muHR

        luminance = (2.0*muHR*muSR + self.C1)/(muHR**2 + muSR**2 + self.C1)
        contrast = (2.0*sigmaHR*sigmaSR + self.C1)/(sigmaHR**2 + sigmaSR**2 + self.C1)
        structure = (2.0*cov + self.C3)/(sigmaHR*sigmaSR + self.C3)

        pcs = tf.math.reduce_prod((contrast**self.beta)*(contrast**self.gamma), axis=0)

        loss = 1 - tf.reduce_sum((luminance**self.alpha)*pcs)/(HR.shape[0]*HR.shape[3])
        if isMixed:
            l1WeightedLoss = tf.reduce_sum(tf.abs(tf.subtract(HR, correctedSR)) * weights)/(HR.shape[0]*HR.shape[3])
            loss = self.eta * loss + (1 - self.eta) * l1WeightedLoss
        return loss

    def computeL1EdgeLoss(self, totalClearPixels, HR, correctedSR):
        l1loss = (1.0 / totalClearPixels) * tf.reduce_sum(tf.abs(tf.subtract(HR, correctedSR)), axis=(1, 2, 3))
        sobelLoss = (1.0 / totalClearPixels) * tf.reduce_sum(tf.abs(tf.subtract(tf.image.sobel_edges(HR),
                                                                                tf.image.sobel_edges(correctedSR))),
                                                             axis=(1, 2, 3, 4))
        return (self.pi * l1loss + (1 - self.pi) * sobelLoss)

    def computeL1Loss(self, totalClearPixels, HR, correctedSR):
        loss = (1.0 / totalClearPixels) * tf.reduce_sum(tf.abs(tf.subtract(HR, correctedSR)), axis=(1, 2, 3))
        return loss

    def computeL2Loss(self, totalClearPixels, HR, correctedSR):
        loss = (1.0 / totalClearPixels) * tf.reduce_sum(tf.square(tf.subtract(HR, correctedSR)), axis=(1, 2, 3))
        return loss

    def computecPSNR(self, totalClearPixels, HR, correctedSR):
        loss = self.computeL2Loss(totalClearPixels, HR, correctedSR)
        # Normalized with respect to bit depth
        cPSNR = 10.0 * (tf.math.log(self.numBytes**2/loss) / tf.math.log(tf.constant(10, dtype=tf.float32)))
        return cPSNR
