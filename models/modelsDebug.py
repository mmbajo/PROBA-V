import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization, InstanceNormalization
from tensorflow_addons.activations import mish
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, Conv2D, Lambda, Add, Reshape

MEAN = 8818.0603   # NIR:8216.2191  # RED: 5376.158
STD = 6534.1132    # NIR:3751.4718  # RED: 4161.090
MAX_SHIFT = 6


class iWDSRConv3D:
    def __init__(self, name, band, mean, std, maxShift):
        self.name = name
        self.band = band
        self.mean = mean
        self.std = std
        self.maxShift = maxShift

    def build(self, scale: int, numFilters: int, kernelSize: tuple,
              numResBlocks: int, expRate: int, decayRate: float,
              numImgLR: int, patchSizeLR: int, isGrayScale: bool) -> Model:
        # Define inputs
        imgLRIn = Input(shape=(patchSizeLR, patchSizeLR, numImgLR, 1)) if isGrayScale \
            else Input(shape=(patchSizeLR, patchSizeLR, numImgLR, 3))

        # Get mean of instance mean patch and over all mean pixel value
        meanImgLR = Lambda(lambda x: tf.reduce_mean(x, axis=3, name='meanLR'), name='getMeanLR')(imgLRIn)

        # Normalize Instance
        imgLR = Lambda(self.normalize, name='normImgLR')(imgLRIn)
        meanImgLR = Lambda(self.normalize, name='normMeanImgLR')(meanImgLR)

        # ImgResBlocks | Main Path
        main = self.iWDSRNetMainPath(imgLR, numFilters, kernelSize,
                                     numResBlocks, patchSizeLR, numImgLR,
                                     scale, expRate, decayRate)

        # MeanResBlocks | Residual Path
        residual = self.iWDSRNetResidualPath(meanImgLR, kernelSize[:-1], scale)

        # Fuse Main and Residual Patch
        out = Add(name='mainPlusResid')([main, residual])

        # Denormalize Instance
        out = Lambda(self.denormalize, name='denorm')(out)

        return Model(imgLRIn, out, name=f'WDSRConv3D_{self.band}_{self.name}')

    def iWDSRNetResidualPath(self, meanImgLR: tf.Tensor, kernelSize: tuple, scale: int):
        x = self.conv2DIns(meanImgLR, outChannels=scale*scale, kernelSize=kernelSize,
                           padding='valid', activation='mish', name='residConv1')
        x = self.conv2DIns(x, outChannels=scale*scale, kernelSize=kernelSize,
                           padding='valid', name='residConv2')
        x = self.conv2DIns(x, outChannels=scale*scale, kernelSize=kernelSize,
                           padding='valid', name='residConv3')
        #  See https://arxiv.org/abs/1609.05158
        x = Lambda(lambda x: tf.nn.depth_to_space(x, scale), name='dtsResid')(x)  # Pixel Shuffle!
        return x

    def iWDSRNetMainPath(self, imgLR: tf.Tensor, numFilters: int, kernelSize: tuple,
                         numResBlocks: int, patchSizeLR: int, numImgLR: int,
                         scale: int, expRate: int, decayRate: int):
        x = self.conv3DIns(imgLR, numFilters, kernelSize, 'same', activation='mish', name='mainConv1')
        for i in range(numResBlocks):
            x = self.ResConv3D(x, numFilters, expRate, decayRate, kernelSize, i)

        x = self.ConvReduceAndUpscale(x, numImgLR, scale, numFilters, kernelSize)
        x = Reshape((patchSizeLR - self.maxShift, patchSizeLR - self.maxShift, scale*scale), name='reshapeMain')(x)
        #  See https://arxiv.org/abs/1609.05158
        x = Lambda(lambda x: tf.nn.depth_to_space(x, scale), name='dtsMain')(x)  # Pixel Shuffle!
        return x

    def ConvReduceAndUpscale(self, x: tf.Tensor, numImgLR: int, scale: int, numFilters: int, kernelSize: tuple):
        '''used in patch 38 numLRImg 9 config'''
        # Conv Reducer
        for i in range(numImgLR//scale):
            if i == 0:
                x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]],
                                            mode='reflect'), name=f'convReducePad_{i}')(x)
            x = self.conv3DIns(x, numFilters, kernelSize, padding='valid',
                               activation='mish', name=f'convReducer_{i}')(x)
        # Upscale block
        x = self.conv3DIns(x, outChannels=scale*scale, kernelSize=kernelSize,
                           padding='valid', name='upscaleConv1')
        return x

    def ConvReduceAndUpscalev2(self, x: tf.Tensor, numImgLR: int, scale: int, numFilters: int, kernelSize: tuple):
        '''used in patch 38 numLRImg 7 config'''
        # Conv Reducer
        for i in range(numImgLR//scale):
            x = self.conv3DIns(x, numFilters, kernelSize, padding='valid',
                               activation='mish', name=f'convReducer_{i}')(x)
        # Upscale block
        x = self.conv3DIns(x, outChannels=scale*scale, kernelSize=kernelSize,
                           padding='valid', name='upscaleConv1')(x)
        return x

    def ResConv3D(self, xIn: tf.Tensor, numFilters: int, expRate: int, decayRate: float, kernelSize: int, blockNum: int):
        # Expansion Conv3d | Same padding
        x = self.conv3DIns(xIn, outChannels=numFilters*expRate, kernelSize=1, padding='same',
                           activation='mish', name=f'expConv_{blockNum}')
        # Decay Conv3d | Same padding
        x = self.conv3DIns(x, xIn=outChannels=int(numFilters*decayRate), kernelSize=1,
                           padding='mish', name=f'decConv_{blockNum}')
        # Norm Conv3D | Same padding
        x = self.conv3DIns(x, outChannels=numFilters, kernelSize=kernelSize,
                           padding='mish', name=f'normConv_{blockNum}')
        # Add input and result
        out = Add(name=f'AddResConv_{blockNum}')([x, xIn])
        return out

    def weightNormedConv3D(self, outChannels: int, kernelSize: int, padding: str, activation=None, name=''):
        return WeightNormalization(Conv3D(outChannels, kernelSize, padding=padding, activation=activation),
                                   data_init=False, name=name)

    def weightNormedConv2D(self, outChannels: int, kernelSize: int, padding: str, activation=None, name=''):
        return WeightNormalization(Conv2D(outChannels, kernelSize, padding=padding, activation=activation),
                                   data_init=False, name=name)

    def conv3DIns(self, xIn, outChannels, kernelSize, padding,  activation=None, name=''):
        if activation is None:
            x = WeightNormalization(Conv3D(outChannels, kernelSize, padding=padding, activation=None),
                                    data_init=False, name=name)(xIn)
            x = InstanceNormalization(axis=4,
                                      center=True,
                                      scale=True,
                                      beta_initializer="random_uniform",
                                      gamma_initializer="random_uniform")(x)
            return x
        if activation == 'leakyrelu':
            x = WeightNormalization(Conv3D(outChannels, kernelSize, padding=padding, activation=None),
                                    data_init=False, name=name)(xIn)
            x = InstanceNormalization(axis=4,
                                      center=True,
                                      scale=True,
                                      beta_initializer="random_uniform",
                                      gamma_initializer="random_uniform")(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
            return x
        if activation == 'mish':
            x = WeightNormalization(Conv3D(outChannels, kernelSize, padding=padding, activation=None),
                                    data_init=False, name=name)(xIn)
            x = InstanceNormalization(axis=4,
                                      center=True,
                                      scale=True,
                                      beta_initializer="random_uniform",
                                      gamma_initializer="random_uniform")(x)
            x = mish()(x)
            return x

    def conv2DIns(self, xIn, outChannels, kernelSize, padding, activation=None, name=''):
        if activation is None:
            x = WeightNormalization(Conv2D(outChannels, kernelSize, padding=padding, activation=None),
                                    data_init=False, name=name)(xIn)
            x = InstanceNormalization(axis=3,
                                      center=True,
                                      scale=True,
                                      beta_initializer="random_uniform",
                                      gamma_initializer="random_uniform")(x)
            return x
        if activation == 'leakyrelu':
            x = WeightNormalization(Conv2D(outChannels, kernelSize, padding=padding, activation=None),
                                    data_init=False, name=name)(xIn)
            x = InstanceNormalization(axis=3,
                                      center=True,
                                      scale=True,
                                      beta_initializer="random_uniform",
                                      gamma_initializer="random_uniform")(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
            return x
        if activation == 'mish':
            x = WeightNormalization(Conv2D(outChannels, kernelSize, padding=padding, activation=None),
                                    data_init=False, name=name)(xIn)
            x = InstanceNormalization(axis=3,
                                      center=True,
                                      scale=True,
                                      beta_initializer="random_uniform",
                                      gamma_initializer="random_uniform")(x)
            x = mish()(x)
            return x

    def normalize(self, x):
        return (x-self.mean)/self.std

    def denormalize(self, x):
        return x * self.std + self.mean
