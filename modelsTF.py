import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization, InstanceNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, Conv2D, Lambda, Add, Reshape

MEAN = 8818.0603
STD = 6534.1132
MAX_SHIFT = 6


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
        imgLRIn = Input(shape=(patchSizeLR, patchSizeLR, numImgLR, 1)) if isGrayScale \
            else Input(shape=(patchSizeLR, patchSizeLR, numImgLR, 3))

        # Get mean of instance mean patch and over all mean pixel value
        meanImgLR = Lambda(lambda x: tf.reduce_mean(x, axis=3, name='meanLR'), name='getMeanLR')(imgLRIn)

        # Normalize Instance
        imgLR = Lambda(self.normalize, name='normImgLR')(imgLRIn)
        meanImgLR = Lambda(self.normalize, name='normMeanImgLR')(meanImgLR)

        # ImgResBlocks | Main Path
        main = self.WDSRNetMainPath(imgLR, numFilters, kernelSize,
                                    numResBlocks, patchSizeLR, numImgLR,
                                    scale, expRate, decayRate)

        # MeanResBlocks | Residual Path
        residual = self.WDSRNetResidualPath(meanImgLR, kernelSize[:-1], scale)

        # Fuse Main and Residual Patch
        out = Add(name='mainPlusResid')([main, residual])

        # Denormalize Instance
        out = Lambda(self.denormalize, name='denorm')(out)

        return Model(imgLRIn, out, name=f'WDSRConv3D_{self.band}_{self.name}')

    def WDSRNetResidualPath(self, meanImgLR: tf.Tensor, kernelSize: tuple, scale: int):
        x = self.weightNormedConv2D(outChannels=scale*scale, kernelSize=kernelSize,
                                    padding='valid', activation='relu', name='residConv1')(meanImgLR)
        x = self.weightNormedConv2D(outChannels=scale*scale, kernelSize=kernelSize,
                                    padding='valid', name='residConv2')(x)
        x = self.weightNormedConv2D(outChannels=scale*scale, kernelSize=kernelSize,
                                    padding='valid', name='residConv3')(x)
        #  See https://arxiv.org/abs/1609.05158
        x = Lambda(lambda x: tf.nn.depth_to_space(x, scale), name='dtsResid')(x)  # Pixel Shuffle!
        return x

    def WDSRNetMainPath(self, imgLR: tf.Tensor, numFilters: int, kernelSize: tuple,
                        numResBlocks: int, patchSizeLR: int, numImgLR: int,
                        scale: int, expRate: int, decayRate: int):
        x = self.weightNormedConv3D(numFilters, kernelSize, 'same', activation='relu', name='mainConv1')(imgLR)
        for i in range(numResBlocks):
            x = self.ResConv3D(x, numFilters, expRate, decayRate, kernelSize, i)

        x = self.ConvReduceAndUpscalev2(x, numImgLR, scale, numFilters, kernelSize)
        x = Reshape((patchSizeLR - self.maxShift, patchSizeLR - self.maxShift, scale*scale), name='reshapeMain')(x)
        #  See https://arxiv.org/abs/1609.05158
        x = Lambda(lambda x: tf.nn.depth_to_space(x, scale), name='dtsMain')(x)  # Pixel Shuffle!
        return x

    def ConvReduceAndUpscale(self, x: tf.Tensor, numImgLR: int, scale: int, numFilters: int, kernelSize: tuple):
        '''used in patch 38 numLRImg 9 config'''
        # Conv Reducer
        for i in range(numImgLR//scale):
            if i == 1:
                x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]],
                                            mode='reflect'), name=f'convReducePad_{i}')(x)
            x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                        activation='relu', name=f'convReducer_{i}')(x)
        # Upscale block
        x = self.weightNormedConv3D(outChannels=scale*scale, kernelSize=kernelSize,
                                    padding='valid', name='upscaleConv1')(x)
        return x

    def ConvReduceAndUpscalev2(self, x: tf.Tensor, numImgLR: int, scale: int, numFilters: int, kernelSize: tuple):
        '''used in patch 38 numLRImg 7 config'''
        # Conv Reducer
        for i in range(numImgLR//scale):
            x = self.weightNormedConv3D(numFilters, kernelSize, padding='valid',
                                        activation='relu', name=f'convReducer_{i}')(x)
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
