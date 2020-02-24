from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D


def weightNormedConv3D(outChannels: int, kernelSize: int, padding: str, activation=None):
    return WeightNormalization(Conv3D(outChannels, kernelSize, padding=padding, activation=activation),
                               data_init=False)
