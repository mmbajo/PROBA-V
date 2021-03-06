from trainClass import *
from utils.loss import *
from utils.utils import *
from modelsTF import *
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Mean
import tensorflow as tf
import numpy as np
import logging
import os
import gc

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('__name__')


def main():

    # import data
    CLEAN_DATA_DIR = '/home/mark/DataBank/PROBA-V-CHKPT/augmentedPatchesDir'
    band = 'NIR'
    X = np.load(os.path.join(CLEAN_DATA_DIR, f'TRAINpatchesLR_{band}.npy'), allow_pickle=True)
    y = np.load(os.path.join(CLEAN_DATA_DIR, f'TRAINpatchesHR_{band}.npy'), allow_pickle=True)

    print(f'Input shape: {X.shape} --------> Output shape: {y.shape}')
    X_train, X_val, y_train, y_val, y_train_mask, y_val_mask = train_test_split(
        X, y, ~y.mask, test_size=0.3, random_state=17)

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    y_train_mask = tf.convert_to_tensor(y_train_mask, dtype=tf.float32)
    y_val_mask = tf.convert_to_tensor(y_val_mask, dtype=tf.float32)

    y = [y_train, y_train_mask]
    valData = [X_val, y_val, y_val_mask]

#    model = WDSRConv3D(scale=3, numFilters=32, kernelSize=(3, 3, 3), numResBlocks=8,
#                       expRate=8, decayRate=0.8, numImgLR=9, patchSizeLR=32, isGrayScale=True)
    with tf.device('/GPU:1'):
        model = WDSRConv3D(scale=3, numFilters=32, kernelSize=(3, 3, 3), numResBlocks=8,
                           expRate=8, decayRate=0.8, numImgLR=9, patchSizeLR=38, isGrayScale=True)
        l = Losses()
        trainClass = ModelTrainer(model=model,
                                  loss=l.shiftCompensatedL1Loss,
                                  metric=l.shiftCompensatedcPSNR,
                                  optimizer=Nadam(learning_rate=5e-4),
                                  ckptDir=f'/home/mark/DataBank/ckpt_{band}_38',
                                  logDir=f'/home/mark/DataBank/logNewRed_{band}_38')
        del X
        gc.collect()

        trainClass.fitTrainData(X_train, y, 64, 10000, 512, valData, 1)


if __name__ == '__main__':
    main()
