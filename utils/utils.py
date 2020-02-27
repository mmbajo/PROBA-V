from typing import List, Tuple

import tensorflow as tf


def samePaddingForConv3d(inputSize: Tuple[int], kernelSize: Tuple[int], stride: Tuple[int]) -> Tuple[int]:
    _, dIn, hIn, wIn = inputSize

    dPad = (((dIn - 1) * stride[0]) - dIn + (kernelSize[0] - 1) + 1) // 2
    hPad = (((hIn - 1) * stride[1]) - hIn + (kernelSize[1] - 1) + 1) // 2
    wPad = (((wIn - 1) * stride[2]) - wIn + (kernelSize[2] - 1) + 1) // 2

    padding = (dPad, hPad, wPad)
    return padding


def loadTrainDataAsTFDataSet(X, y, epochs, batchSize, bufferSize):
    return tf.data.Dataset.from_tensor_slices(
        (X, y, y.mask)).shuffle(bufferSize, reshuffle_each_iteration=True).repeat(epochs).batch(batchSize).prefetch(tf.data.experimental.AUTOTUNE)


def loadValDataAsTFDataSet(X, y, valSteps, batchSize, bufferSize):
    return tf.data.Dataset.from_tensor_slices(
        (X, y, y.mask)).shuffle(bufferSize).batch(batchSize).prefetch(tf.data.experimental.AUTOTUNE).take(valSteps)
