from typing import List, Dict, Tuple

import argparse
import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD, Nadam

from modelsTF import WDSRConv3D
from trainClass import ModelTrainer
from utils.loss import Losses
from utils.utils import *

import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('__name__')


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/mark/DataBank/PROBA-V-CHKPT/augmentedPatchesDir')
    parser.add_argument('--band', type=str, default='NIR')
    parser.add_argument('--split', type=float, default=0.3)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--logDir', type=str, default='modelInfo/logs_38_top7_90p')
    parser.add_argument('--ckptDir', type=str, default='modelInfo/ckpt_38_top7_90p')
    parser.add_argument('--optim', type=str, default='nadam')
    opt = parser.parse_args()
    return opt


def main():
    logger.info('[ INFO ] Loading data...')

    patchHR = np.load(os.path.join(opt.data, f'TRAINpatchesHR_{opt.band}.npy'), allow_pickle=True)
    patchLR = np.load(os.path.join(opt.data, f'TRAINpatchesLR_{opt.band}.npy'), allow_pickle=True)

    logger.info('[ INFO ] Computing data stats...')
    datasetAllMean = 8818.0603
    datasetAllStd = 6534.1132

    logger.info('[ INFO ] Splitting data...')
    X_train, X_val, y_train, y_val, y_train_mask, y_val_mask = train_test_split(
        patchLR, patchHR, ~patchHR.mask, test_size=opt.split, random_state=17)

    logger.info('[ INFO ] Freeing up memory...')
    del patchLR
    del patchHR
    gc.collect()

    logger.info('[ INFO ] Converting masked_array to array...')
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_train_mask = np.array(y_train_mask)
    y_val_mask = np.array(y_val_mask)

    y = [y_train, y_train_mask]
    valData = [X_val, y_val, y_val_mask]

    logger.info('[ INFO ] Instantiate model...')
    modelIns = WDSRConv3D(name='patch38', band=opt.band, mean=datasetAllMean, std=datasetAllStd, maxShift=6)
    logger.info('[ INFO ] Building model...')
    model = modelIns.build(scale=3, numFilters=32, kernelSize=(3, 3, 3), numResBlocks=8,
                           expRate=8, decayRate=0.8, numImgLR=7, patchSizeLR=38, isGrayScale=True)

    logger.info(f'[ INFO ] Initialize {opt.optim.upper()} optimizer...')
    if opt.optim == 'adam':
        optimizer = Adam(learning_rate=5e-4)
    elif opt.optim == 'nadam':
        # http://cs229.stanford.edu/proj2015/054_report.pdf
        optimizer = Nadam(learning_rate=5e-4)
    else:
        optimizer = SGD(learning_rate=5e-4)

    logger.info('[ INFO ] Initialize Trainer...')
    loss = Losses()
    ckptDir = os.path.join(opt.ckptDir, opt.band)
    logDir = os.path.join(opt.logDir, opt.band)
    trainClass = ModelTrainer(model=model,
                              loss=loss.shiftCompensatedL1Loss,
                              metric=loss.shiftCompensatedcPSNR,
                              optimizer=optimizer,
                              ckptDir=ckptDir,
                              logDir=logDir)

    trainClass.fitTrainData(X_train, y, opt.batchSize, opt.epochs, valData)

    logger.info(f'[ SUCCESS ] Model checkpoint can be found in {ckptDir}.')
    logger.info(f'[ SUCCESS ] Model logs can be found in {logDir}.')


if __name__ == '__main__':
    opt = parser()
    main()
