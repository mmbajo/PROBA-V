import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization, InstanceNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, Conv2D, Lambda, Add, Reshape


class WDSRConv3D:
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
        imgLRIn = Input(shape=(patchSizeLR + self.maxShift, patchSizeLR + self.maxShift, numImgLR, 1)) if isGrayScale \
            else Input(shape=(patchSizeLR + self.maxShift, patchSizeLR + self.maxShift, numImgLR, 3))

        # Get mean of instance mean patch and over all mean pixel value
        meanImgLR = Lambda(lambda x: tf.reduce_mean(x, axis=3, name='meanLR'), name='getMeanLR')(imgLRIn)

        # Normalize Instance
        imgLR = Lambda(self.normalize, name='normImgLR')(imgLRIn)
        meanImgLR = Lambda(self.normalize, name='normMeanImgLR')(meanImgLR)

        # ImgResBlocks | High Frequency Residuals Path
        main = self.WDSRNetHRResidualPath(imgLR, numFilters, kernelSize,
                                          numResBlocks, patchSizeLR, numImgLR,
                                          scale, expRate, decayRate)

        # MeanResBlocks | Low Frequency Residuals Path
        residual = self.WDSRNetLRResidualPath(meanImgLR, kernelSize[:-1], scale)

        # Fuse Main and Residual Patch
        out = Add(name='mainPlusResid')([main, residual])

        # Denormalize Instance
        out = Lambda(self.denormalize, name='denorm')(out)

        return Model(imgLRIn, out, name=f'WDSRConv3D_{self.band}_{self.name}')

    def WDSRNetLRResidualPath(self, x: tf.Tensor, kernelSize: tuple, scale: int):
        # TODO: Check correctness for different scales
        for i in range(scale):
            act = 'relu' if i == 0 else None
            x = self.weightNormedConv2D(outChannels=scale*scale, kernelSize=kernelSize,
                                        padding='valid', activation=act, name=f'residConv{i+1}')(x)
        #  See https://arxiv.org/abs/1609.05158
        x = Lambda(lambda x: tf.nn.depth_to_space(x, scale), name='dtsResid')(x)  # Pixel Shuffle!
        return x

    def WDSRNetHRResidualPath(self, imgLR: tf.Tensor, numFilters: int, kernelSize: tuple,
                              numResBlocks: int, patchSizeLR: int, numImgLR: int,
                              scale: int, expRate: int, decayRate: int):
        x = self.weightNormedConv3D(numFilters, kernelSize, 'same', activation='relu', name='mainConv1')(imgLR)
        for i in range(numResBlocks):
            x = self.ResConv3D(x, numFilters, expRate, decayRate, kernelSize, i)

        if numImgLR == 7:
            x = self.ConvReduceAndUpscalev2(x, numImgLR, scale, numFilters, kernelSize)
        elif numImgLR == 9:
            x = self.ConvReduceAndUpscale(x, numImgLR, scale, numFilters, kernelSize)
        elif numImgLR == 13:
            x = self.ConvReduceAndUpscalev3(x, numImgLR, scale, numFilters, kernelSize)
        elif numImgLR == 19:
            x = self.ConvReduceAndUpscaleEx(x, numImgLR, scale, numFilters, kernelSize)

        x = Reshape((patchSizeLR, patchSizeLR, scale*scale), name='reshapeMain')(x)
        #  See https://arxiv.org/abs/1609.05158
        x = Lambda(lambda x: tf.nn.depth_to_space(x, scale), name='dtsMain')(x)  # Pixel Shuffle!
        return x

    def ConvReduceAndUpscaleEx(self, x: tf.Tensor, numImgLR: int, scale: int, numFilters: int, kernelSize: tuple):
        '''EXPERIMENTAL'''
        x = Lambda(lambda x: tf.pad(x, [[0, 0], [2, 2], [2, 2], [2, 2], [0, 0]],
                                    mode='reflect'), name=f'convReducePad_{1}')(x)
        x = self.weightNormedConv3D(numFilters, (5, 5, 5), padding='valid',
                                    activation='relu', name=f'convReducer_{1}')(x)

        x = Lambda(lambda x: tf.pad(x, [[0, 0], [2, 2], [2, 2], [1, 1], [0, 0]],
                                    mode='reflect'), name=f'convReducePad_{2}')(x)
        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{2}')(x)

        x = Lambda(lambda x: tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0], [0, 0]],
                                    mode='reflect'), name=f'convReducePad_{3}')(x)
        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{3}')(x)

        x = Lambda(lambda x: tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0], [0, 0]],
                                    mode='reflect'), name=f'convReducePad_{4}')(x)
        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{4}')(x)

        x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]],
                                    mode='reflect'), name=f'convReducePad_{5}')(x)
        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{5}')(x)

        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{6}')(x)

        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{7}')(x)

        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{8}')(x)

        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{9}')(x)

        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{10}')(x)

        # Upscale block
        x = self.weightNormedConv3D(outChannels=scale*scale, kernelSize=kernelSize,
                                    padding='valid', name='upscaleConv1')(x)
        return x

    def ConvReduceAndUpscalev3(self, x: tf.Tensor, numImgLR: int, scale: int, numFilters: int, kernelSize: tuple):
        '''used numLRImg 13 config'''
        # Conv Reducer
        x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]],
                                    mode='reflect'), name=f'convReducePad_{1}')(x)
        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{1}')(x)

        x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]],
                                    mode='reflect'), name=f'convReducePad_{2}')(x)
        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{2}')(x)

        x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]],
                                    mode='reflect'), name=f'convReducePad_{3}')(x)
        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{3}')(x)

        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{4}')(x)

        x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                    activation='relu', name=f'convReducer_{5}')(x)

        # Upscale block
        x = self.weightNormedConv3D(outChannels=scale*scale, kernelSize=kernelSize,
                                    padding='valid', name='upscaleConv1')(x)
        return x

    def ConvReduceAndUpscale(self, x: tf.Tensor, numImgLR: int, scale: int, numFilters: int, kernelSize: tuple):
        '''used in patch 38 numLRImg 9 config'''
        # Conv Reducer
        for i in range(numImgLR//scale):
            if i == 0:
                x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]],
                                            mode='reflect'), name=f'convReducePad_{i+1}')(x)
            x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                        activation='relu', name=f'convReducer_{i+1}')(x)
        # Upscale block
        x = self.weightNormedConv3D(outChannels=scale*scale, kernelSize=kernelSize,
                                    padding='valid', name='upscaleConv1')(x)
        return x

    def ConvReduceAndUpscalev2(self, x: tf.Tensor, numImgLR: int, scale: int, numFilters: int, kernelSize: tuple):
        '''used in patch 38 numLRImg 7 config'''
        # Conv Reducer
        for i in range(numImgLR//scale):
            x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                        activation='relu', name=f'convReducer_{i+1}')(x)
        # Upscale block
        x = self.weightNormedConv3D(outChannels=scale*scale, kernelSize=kernelSize,
                                    padding='valid', name='upscaleConv1')(x)
        return x

    def ResConv3D(self, xIn: tf.Tensor, numFilters: int, expRate: int, decayRate: float, kernelSize: int, blockNum: int):
        # Expansion Conv3d | Same padding
        x = self.weightNormedConv3D(outChannels=numFilters*expRate, kernelSize=1, padding='same',
                                    activation='relu', name=f'expConv_{blockNum}')(xIn)
        # Decay Conv3d | Same padding
        x = self.weightNormedConv3D(outChannels=int(numFilters*decayRate), kernelSize=1,
                                    padding='same', name=f'decConv_{blockNum}')(x)
        # Norm Conv3D | Same padding
        x = self.weightNormedConv3D(outChannels=numFilters, kernelSize=kernelSize,
                                    padding='same', name=f'normConv_{blockNum}')(x)
        # Add input and result
        out = Add(name=f'AddResConv_{blockNum}')([x, xIn])
        return out

    def weightNormedConv3D(self, outChannels: int, kernelSize: int, padding: str, activation=None, name=''):
        return WeightNormalization(Conv3D(outChannels, kernelSize, padding=padding, activation=activation),
                                   data_init=False, name=name)

    def weightNormedConv2D(self, outChannels: int, kernelSize: int, padding: str, activation=None, name=''):
        return WeightNormalization(Conv2D(outChannels, kernelSize, padding=padding, activation=activation),
                                   data_init=False, name=name)

    def normalize(self, x):
        return (x-self.mean)/self.std

    def denormalize(self, x):
        return x * self.std + self.mean


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
        imgLRIn = Input(shape=(patchSizeLR + self.maxShift, patchSizeLR + self.maxShift, numImgLR, 1)) if isGrayScale \
            else Input(shape=(patchSizeLR + self.maxShift, patchSizeLR + self.maxShift, numImgLR, 3))

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

    def iWDSRNetResidualPath(self, x: tf.Tensor, kernelSize: tuple, scale: int):
        x = self.conv2DIns(x, outChannels=scale*scale, kernelSize=kernelSize,
                           padding='valid', activation='mish', name='residConv1')
        x = self.conv2DIns(x, outChannels=scale*scale, kernelSize=kernelSize,
                           padding='valid', name='residConv2')
        x = self.conv2DIns(x, outChannels=scale*scale, kernelSize=kernelSize,
                           padding='valid', name='residConv3')
        for i in range(scale):
            act = 'mish' if i == 0 else None
            x = self.conv2DIns(x, outChannels=scale*scale, kernelSize=kernelSize,
                               padding='valid', activation=act, name=f'residConv{i+1}')
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
        x = Reshape((patchSizeLR, patchSizeLR, scale*scale), name='reshapeMain')(x)
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
                               activation='mish', name=f'convReducer_{i}')
        # Upscale block
        x = self.conv3DIns(x, outChannels=scale*scale, kernelSize=kernelSize,
                           padding='valid', name='upscaleConv1')
        return x

    def ConvReduceAndUpscalev2(self, x: tf.Tensor, numImgLR: int, scale: int, numFilters: int, kernelSize: tuple):
        '''used in patch 38 numLRImg 7 config'''
        # Conv Reducer
        for i in range(numImgLR//scale):
            x = self.conv3DIns(x, numFilters, kernelSize, padding='valid',
                               activation='mish', name=f'convReducer_{i}')
        # Upscale block
        x = self.conv3DIns(x, outChannels=scale*scale, kernelSize=kernelSize,
                           padding='valid', name='upscaleConv1')
        return x

    def ResConv3D(self, xIn: tf.Tensor, numFilters: int, expRate: int, decayRate: float, kernelSize: int, blockNum: int):
        # Expansion Conv3d | Same padding
        x = self.conv3DIns(xIn, outChannels=numFilters*expRate, kernelSize=1, padding='same',
                           activation='mish', name=f'expConv_{blockNum}')
        # Decay Conv3d | Same padding
        x = self.conv3DIns(x, outChannels=int(numFilters*decayRate), kernelSize=1,
                           padding='same', name=f'decConv_{blockNum}')
        # Norm Conv3D | Same padding
        x = self.conv3DIns(x, outChannels=numFilters, kernelSize=kernelSize,
                           padding='same', name=f'normConv_{blockNum}')
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
            x = self.mish(x)
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
            x = self.mish(x)
            return x

    def mish(self, x):
        return x * tf.math.tanh(tf.keras.activations.softplus(x))

    def normalize(self, x):
        return (x-self.mean)/self.std

    def denormalize(self, x):
        return x * self.std + self.mean


class FuseNetConv2D:
    def __init__(self, name, band):
        self.name = name
        self.band = band

    def build(self) -> Model:
        # Define inputs
        imgLRIn = Input(shape=(384, 384, 1))

        # Fusing patch
        main = self.FuseNetv3(imgLRIn)

        # Fuse Main and Residual Patch
        out = Add(name='mainPlusInput')([imgLRIn, main])

        return Model(imgLRIn, out, name=f'FuseNet_{self.band}_{self.name}')

    def FuseNet(self, xIn):
        x = Conv2D(128, 3, 3, padding='same')(xIn)
        x = InstanceNormalization(axis=3,
                                  center=True,
                                  scale=True,
                                  beta_initializer="random_uniform",
                                  gamma_initializer="random_uniform")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

        x = Conv2D(64, 3, 1, padding='same')(x)
        x = InstanceNormalization(axis=3,
                                  center=True,
                                  scale=True,
                                  beta_initializer="random_uniform",
                                  gamma_initializer="random_uniform")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

        x = Conv2D(32, 3, 1, padding='same')(x)
        x = InstanceNormalization(axis=3,
                                  center=True,
                                  scale=True,
                                  beta_initializer="random_uniform",
                                  gamma_initializer="random_uniform")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

        x = Conv2D(9, 3, 1, padding='same')(x)
        x = InstanceNormalization(axis=3,
                                  center=True,
                                  scale=True,
                                  beta_initializer="random_uniform",
                                  gamma_initializer="random_uniform")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        x = Lambda(lambda x: tf.nn.depth_to_space(x, 3), name='dtsMain2')(x)

        return x

    def FuseNetv2(self, xIn):
        x = Conv2D(64, 8, 8, padding='same')(xIn)
        x = InstanceNormalization(axis=3,
                                  center=True,
                                  scale=True,
                                  beta_initializer="random_uniform",
                                  gamma_initializer="random_uniform")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

        x = Conv2D(64, 3, 1, padding='same')(x)
        x = InstanceNormalization(axis=3,
                                  center=True,
                                  scale=True,
                                  beta_initializer="random_uniform",
                                  gamma_initializer="random_uniform")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        x = Lambda(lambda x: tf.nn.depth_to_space(x, 8), name='dtsMain2')(x)

        return x

    def FuseNetv3(self, xIn):
        x = Conv2D(64, 48, 1, padding='same')(xIn)
        x = InstanceNormalization(axis=3,
                                  center=True,
                                  scale=True,
                                  beta_initializer="random_uniform",
                                  gamma_initializer="random_uniform")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        x = Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True), name='mean')(x)

        return x
