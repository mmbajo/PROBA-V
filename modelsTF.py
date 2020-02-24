import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, Conv2D, Lambda, Add, Reshape


def WDSRConv3D(scale: int, numFilters: int, resBlockConv3D: tf.layers, kernelSize: tuple,
               numResBlocks: int, expRate: int, decayRate: float, numImgLR: int, patchSizeLR: int) -> Model:
    # Define inputs
    imgLR = Input(shape=(patchSizeLR, patchSizeLR, numImgLR, 1))

    # Get mean of instance mean patch and over all mean pixel value
    meanImgLR = Lambda(lambda x: tf.reduce_mean(x, axis=2))(imgLR)
    allMean = Lambda(lambda x: tf.reduce_mean(x))(imgLR)
    allStdDev = Lambda(lambda x: tf.reduce_std(x))(imgLR)

    # Normalize Instance
    imgLR = Lambda(lambda x: tf.math.divide(tf.math.subtract(x, allMean), allStdDev))(imgLR)
    meanImgLR = Lambda(lambda x: tf.math.divide(tf.math.subtract(x, allMean), allStdDev))(meanImgLR)

    # ImgResBlocks | Main Path
    main = WDSRNetMainPath(imgLR, numFilters, kernelSize,
                           numResBlocks, patchSizeLR, numImgLR,
                           scale, expRate, decayRate)

    # MeanResBlocks | Residual Path
    residual = WDSRNetResidualPath(meanImgLR, scale)

    # Fuse Main and Residual Patch
    out = Add()([main, residual])

    # Denormalize Instance
    out = Lambda(lambda x: tf.math.add(tf.math.multiply(x, allStdDev), allMean))

    return Model(imgLR, out, name='WDSRConv3D')


def WDSRNetResidualPath(meanImgLR: tf.tensor, kernelSize: tuple, scale: int):
    x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT'))(meanImgLR)
    x = weightNormedConv2D(outChannels=scale*scale, kernelSize=kernelSize, padding='valid', activation='relu')(x)
    x = weightNormedConv2D(outChannels=scale*scale, kernelSize=kernelSize, padding='valid')(x)


def WDSRNetMainPath(imgLR: tf.tensor, numFilters: int, kernelSize: tuple,
                    numResBlocks: int, patchSizeLR: int, numImgLR: int,
                    scale: int, expRate: int, decayRate: int):
    x = weightNormedConv3D(numFilters, kernelSize, 'same', activation='relu')(imgLR)
    for _ in range(numResBlocks):
        x = ResConv3D(x, numFilters, expRate, decayRate, kernelSize)

    x = ConvReduceAndUpscale(x, numImgLR, scale, numFilters)
    x = Reshape((patchSizeLR, patchSizeLR, scale*scale))(x)
    x = Lambda(lambda x: tf.nn.depth_to_space(x, scale))(x)
    return x


def ConvReduceAndUpscale(x: tf.tensor, numImgLR: int, scale: int, numFilters: int, kernelSize: tuple):
    # Conv Reducer
    for _ in range(numImgLR//scale):
        x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]], mode='reflect'))(x)
        x = weightNormedConv3D(numFilters, kernelSize, padding='valid', activation='relu')(x)
    # Upscale block
    x = weightNormedConv3D(outChannels=scale*scale, kernelSize=kernelSize, padding='valid')(x)


def ResConv3D(xIn: tf.tensor, numFilters: int, expRate: int, decayRate: float, kernelSize: int):
    # Expansion Conv3d | Same padding
    x = weightNormedConv3D(outChannels=numFilters*expRate, kernelSize=1, padding='same', activation='relu')(xIn)
    # Decay Conv3d | Same padding
    x = weightNormedConv3D(outChannels=numFilters*decayRate, kernelSize=1, padding='same')(x)
    # Norm Conv3D | Same padding
    x = weightNormedConv3D(outChannels=numFilters, kernelSize=kernelSize, padding='same')(x)
    # Add input and result
    out = Add([x, xIn])
    return out


def weightNormedConv3D(outChannels: int, kernelSize: int, padding: str, activation=None):
    return WeightNormalization(Conv3D(outChannels, kernelSize, padding=padding, activation=activation),
                               data_init=False)


def weightNormedConv2D(outChannels: int, kernelSize: int, padding: str, activation=None):
    return WeightNormalization(Conv2D(outChannels, kernelSize, padding=padding, activation=activation),
                               data_init=False)
