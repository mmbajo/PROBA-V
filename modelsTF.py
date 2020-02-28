import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, Conv2D, Lambda, Add, Reshape

MEAN = 7433.6436
STD = 2353.0723


def WDSRConv3D(scale: int, numFilters: int, kernelSize: tuple,
               numResBlocks: int, expRate: int, decayRate: float,
               numImgLR: int, patchSizeLR: int, isGrayScale: bool) -> Model:
    # Define inputs
    imgLRIn = Input(shape=(patchSizeLR, patchSizeLR, numImgLR, 1)) if isGrayScale \
        else Input(shape=(patchSizeLR, patchSizeLR, numImgLR, 3))

    # Get mean of instance mean patch and over all mean pixel value
    imgLR = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]], mode='reflect'), name='initPad')(imgLRIn)
    meanImgLR = Lambda(lambda x: tf.reduce_mean(x, axis=3, name='meanLR'), name='getMeanLR')(imgLR)
    #allMean = Lambda(lambda x: tf.reduce_mean(x, name='allMean'), name='getAllMean')(imgLR)
    #allStdDev = Lambda(lambda x: tf.math.reduce_std(x, name='allStdDev'), name='getAllStdDev')(imgLR)

    # Normalize Instance
    imgLR = Lambda(normalize, name='normImgLR')(imgLR)
    meanImgLR = Lambda(normalize, name='normMeanImgLR')(meanImgLR)

    # ImgResBlocks | Main Path
    main = WDSRNetMainPath(imgLR, numFilters, kernelSize,
                           numResBlocks, patchSizeLR, numImgLR,
                           scale, expRate, decayRate)

    # MeanResBlocks | Residual Path
    residual = WDSRNetResidualPath(meanImgLR, kernelSize[:-1], scale)

    # Fuse Main and Residual Patch
    out = Add(name='mainPlusResid')([main, residual])

    # Denormalize Instance
    out = Lambda(denormalize, name='denorm')(out)

    return Model(imgLRIn, out, name='WDSRConv3D')


def WDSRNetResidualPath(meanImgLR: tf.Tensor, kernelSize: tuple, scale: int):
    x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT'), name='padResid')(meanImgLR)
    x = weightNormedConv2D(outChannels=scale*scale, kernelSize=kernelSize, padding='valid', activation='relu', name='residConv1')(x)
    x = weightNormedConv2D(outChannels=scale*scale, kernelSize=kernelSize, padding='valid', name='residConv2')(x)
    x = Lambda(lambda x: tf.nn.depth_to_space(x, scale), name='dtsResid')(x)
    return x


def WDSRNetMainPath(imgLR: tf.Tensor, numFilters: int, kernelSize: tuple,
                    numResBlocks: int, patchSizeLR: int, numImgLR: int,
                    scale: int, expRate: int, decayRate: int):
    x = weightNormedConv3D(numFilters, kernelSize, 'same', activation='relu', name='mainConv1')(imgLR)
    for i in range(numResBlocks):
        x = ResConv3D(x, numFilters, expRate, decayRate, kernelSize, i)

    x = ConvReduceAndUpscale(x, numImgLR, scale, numFilters, kernelSize)
    x = Reshape((patchSizeLR, patchSizeLR, scale*scale), name='reshapeMain')(x)
    x = Lambda(lambda x: tf.nn.depth_to_space(x, scale), name='dtsMain')(x)
    return x


def ConvReduceAndUpscale(x: tf.Tensor, numImgLR: int, scale: int, numFilters: int, kernelSize: tuple):
    # Conv Reducer
    for i in range(numImgLR//scale):
        x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]], mode='reflect'), name=f'convReducePad_{i}')(x)
        x = weightNormedConv3D(numFilters, kernelSize, padding='valid', activation='relu', name=f'convReducer_{i}')(x)
    # Upscale block
    x = weightNormedConv3D(outChannels=scale*scale, kernelSize=kernelSize, padding='valid', name='upscaleConv')(x)
    return x


def ResConv3D(xIn: tf.Tensor, numFilters: int, expRate: int, decayRate: float, kernelSize: int, blockNum: int):
    # Expansion Conv3d | Same padding
    x = weightNormedConv3D(outChannels=numFilters*expRate, kernelSize=1, padding='same', activation='relu', name=f'expConv_{blockNum}')(xIn)
    # Decay Conv3d | Same padding
    x = weightNormedConv3D(outChannels=int(numFilters*decayRate), kernelSize=1, padding='same', name=f'decConv_{blockNum}')(x)
    # Norm Conv3D | Same padding
    x = weightNormedConv3D(outChannels=numFilters, kernelSize=kernelSize, padding='same', name=f'normConv_{blockNum}')(x)
    # Add input and result
    out = Add(name=f'AddResConv_{blockNum}')([x, xIn])
    return out


def weightNormedConv3D(outChannels: int, kernelSize: int, padding: str, activation=None, name=''):
    return WeightNormalization(Conv3D(outChannels, kernelSize, padding=padding, activation=activation),
                               data_init=False, name=name)


def weightNormedConv2D(outChannels: int, kernelSize: int, padding: str, activation=None, name=''):
    return WeightNormalization(Conv2D(outChannels, kernelSize, padding=padding, activation=activation),
                               data_init=False, name=name)

def normalize(x):
    return (x-MEAN)/STD

def denormalize(x):
    return x * STD + MEAN 