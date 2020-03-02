import tensorflow as tf
import numpy as np
from tqdm import tqdm

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logger = logging.getLogger('__name__')


class Enhancer:
    def __init__(self, model, patchLR):
        self.model = model
        self.patchLR = patchLR

    def enhance(self):
        enhancedImgs = []
        for set in tqdm(self.patchLR):
            enhancedPatches = self.enhancePatch(set)
            enhancedImg = self.reconstruct(np.array(enhancedPatches))
            enhancedImgs.append(enhancedImg)
        return enhancedImgs

    def enhancePatch(self, set):
        lr_batch = tf.cast(set, tf.float32)
        sr_batch = self.model(lr_batch)
        sr_batch = tf.clip_by_value(sr_batch, 0, 2**16)
        sr_batch = tf.round(sr_batch)
        sr_batch = tf.cast(sr_batch, tf.float32)
        return sr_batch

    def reconstruct(self, patches):
        img = np.zeros((384, 384, 1))
        block_n = 0
        for i in range(1, 5):
            for j in range(1, 5):
                img[(i-1)*96:i*96, (j-1)*96:j*96] = patches[block_n, :, :, ]
                block_n += 1
        return img.reshape((384, 384, 1))
