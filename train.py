from typing import List, Dict, Tuple

import argparse
import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD, Nadam

from models.modelsTF import WDSRConv3D, iWDSRConv3D, FuseNetConv2D
from models.trainClass import ModelTrainer
from models.loss import Losses
from utils.utils import *
from tqdm import tqdm
from skimage import io

import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('__name__')


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/mark/DataBank/PROBA-V-CHKPT/augmentedPatchesDir')
    parser.add_argument('--band', type=str, default='NIR')
    parser.add_argument('--split', type=float, default=0.2)
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--logDir', type=str, default='modelInfo/logs_16_top9_85p_12Res_32_L1Loss')
    parser.add_argument('--ckptDir', type=str, default='modelInfo/ckpt_16_top9_85p_12Res_32_L1Loss')
    parser.add_argument('--optim', type=str, default='nadam')
    parser.add_argument('--modelType', type=str, default='patchNet')
    opt = parser.parse_args()
    return opt


def patchNet():
    logger.info('[ INFO ] Loading data...')

    patchHR = np.load(os.path.join(opt.data, f'TRAINpatchesHR_{opt.band}.npy'), allow_pickle=True)
    patchLR = np.load(os.path.join(opt.data, f'TRAINpatchesLR_{opt.band}.npy'), allow_pickle=True)

    logger.info('[ INFO ] Computing data stats...')
    if opt.band == 'NIR':
        datasetAllMean = 8075.2045  # 8818.0603
        datasetAllStd = 3160.7272  # 6534.1132
    else:
        datasetAllMean = 5266.2245
        datasetAllStd = 3431.8614

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
    modelIns = WDSRConv3D(name='patch38_8ResBlocks', band=opt.band,
                          mean=datasetAllMean, std=datasetAllStd, maxShift=6)
    logger.info('[ INFO ] Building model...')
    model = modelIns.build(scale=3, numFilters=32, kernelSize=(3, 3, 3), numResBlocks=12,
                           expRate=8, decayRate=0.8, numImgLR=9, patchSizeLR=16, isGrayScale=True)

    logger.info(f'[ INFO ] Initialize {opt.optim.upper()} optimizer...')
    if opt.optim == 'adam':
        optimizer = Adam(learning_rate=5e-4)
    elif opt.optim == 'nadam':
        # http://cs229.stanford.edu/proj2015/054_report.pdf
        optimizer = Nadam(learning_rate=5e-4)
    else:
        optimizer = SGD(learning_rate=5e-4)

    logger.info('[ INFO ] Initialize Trainer...')
    loss = Losses(targetShape=(48, 48, 1))
    ckptDir = os.path.join(opt.ckptDir, opt.band)
    logDir = os.path.join(opt.logDir, opt.band)
    trainClass = ModelTrainer(model=model,
                              loss=loss.shiftCompensatedL1Loss,  # ,shiftCompensatedL1EdgeLoss
                              metric=loss.shiftCompensatedcPSNR,
                              optimizer=optimizer,
                              ckptDir=ckptDir,
                              logDir=logDir)

    trainClass.fitTrainData(X_train, y, opt.batchSize, opt.epochs, valData)

    logger.info(f'[ SUCCESS ] Model checkpoint can be found in {ckptDir}.')
    logger.info(f'[ SUCCESS ] Model logs can be found in {logDir}.')


def fusionNet():
    # Input data
    logger.info('[ INFO ] Loading SR data...')
    # opt.fusionDataPath
    fusionedImDir = '/home/mark/DataBank/PROBA-V-CHKPT/old/results/testout_patch38_top9_85p_12res_L1Loss'
    imageNames = sorted(os.listdir(fusionedImDir))
    images = []
    for i, name in tqdm(enumerate(imageNames), total=1160):
        if i == 1160:
            break
        img = io.imread(os.path.join(fusionedImDir, name))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        images.append(img)
    inputImgs = np.concatenate(images)

    # Load the ground truth imageSets
    logger.info('[ INFO ] Loading HR data...')
    dirName = '/home/mark/DataBank/PROBA-V-CHKPT/trimmedArrayDir'
    red = 'TRAINimgHR_RED.npy'
    nir = 'TRAINimgHR_NIR.npy'
    red = np.load(os.path.join(dirName, red), allow_pickle=True)
    nir = np.load(os.path.join(dirName, nir), allow_pickle=True)
    allImgMsk = np.ma.concatenate((red, nir))
    allImgMsk = allImgMsk.squeeze(1)
    allImgMsk = allImgMsk.astype(np.float32)
    allImgMsk = allImgMsk.transpose((0, 2, 3, 1))

    # Split data
    logger.info('[ INFO ] Splitting data...')
    X_train, X_val, y_train, y_val, y_train_mask, y_val_mask = train_test_split(
        inputImgs, allImgMsk, ~allImgMsk.mask, test_size=opt.split, random_state=17)

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
    modelIns = FuseNetConv2D(name='fuseme', band=opt.band)
    model = modelIns.build()

    logger.info(f'[ INFO ] Initialize {opt.optim.upper()} optimizer...')
    if opt.optim == 'adam':
        optimizer = Adam(learning_rate=5e-4)
    elif opt.optim == 'nadam':
        # http://cs229.stanford.edu/proj2015/054_report.pdf
        optimizer = Nadam(learning_rate=5e-4)
    else:
        optimizer = SGD(learning_rate=5e-4)

    logger.info('[ INFO ] Initialize Trainer...')
    loss = Losses(targetShape=(384, 384, 1))
    ckptDir = 'fuseNetCkpt'
    logDir = 'fuseNetLogs'
    trainClass = ModelTrainer(model=model,
                              loss=loss.shiftCompensatedL1Loss,  # ,shiftCompensatedL1EdgeLoss
                              metric=loss.shiftCompensatedcPSNR,
                              optimizer=optimizer,
                              ckptDir=ckptDir,
                              logDir=logDir)

    trainClass.fitTrainData(X_train, y, opt.batchSize, opt.epochs, valData)

    logger.info(f'[ SUCCESS ] Model checkpoint can be found in {ckptDir}.')
    logger.info(f'[ SUCCESS ] Model logs can be found in {logDir}.')


if __name__ == '__main__':
    opt = parser()
    if opt.modelType == 'patchNet':
        patchNet()
    else:
        fusionNet()
