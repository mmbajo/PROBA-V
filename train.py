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
from utils.parseConfig import parseConfig
from utils.utils import *
from tqdm import tqdm
from skimage import io

import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('__name__')


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfg/yourcfg.cfg', type=str)
    parser.add_argument('--band', type=str, default='NIR')
    parser.add_argument('--modelType', type=str, default='patchNet')
    opt = parser.parse_args()
    return opt


def patchNet(config):
    logger.info('[ INFO ] Loading data...')
    dataDir = os.path.join(config['preprocessing_out'], 'augmentedPatchesDir')

    X_train = np.load(os.path.join(dataDir, f'TRAINpatchesLR_{opt.band}.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(dataDir, f'TRAINVALpatchesLR_{opt.band}.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(dataDir, f'TRAINpatchesHR_{opt.band}.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(dataDir, f'TRAINVALpatchesHR_{opt.band}.npy'), allow_pickle=True)
    y_train_mask = ~y_train.mask
    y_val_mask = ~y_val.mask

    logger.info('[ INFO ] Loading data stats...')
    if opt.band == 'NIR':
        datasetAllMean = 8075.2045  # 8818.0603
        datasetAllStd = 3160.7272  # 6534.1132
    else:
        datasetAllMean = 5266.2245
        datasetAllStd = 3431.8614

    logger.info('[ INFO ] Converting masked array to array...')
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_train_mask = np.array(y_train_mask)
    y_val_mask = np.array(y_val_mask)

    y = [y_train, y_train_mask]
    valData = [X_val, y_val, y_val_mask]

    logger.info('[ INFO ] Instantiate model...')
    modelIns = WDSRConv3D(name='superResolutionNet', band=opt.band,
                          mean=datasetAllMean, std=datasetAllStd, maxShift=config['max_shift'])

    logger.info('[ INFO ] Building model...')
    kernelSize = (config['kernel_size'], config['kernel_size'], config['kernel_size'])
    model = modelIns.build(scale=config['scale'], numFilters=config['num_filters'], kernelSize=kernelSize,
                           numResBlocks=config['num_res_blocks'], expRate=config['exp_rate'],
                           decayRate=config['decay_rate'], numImgLR=config['num_low_res_imgs'],
                           patchSizeLR=config['patch_size'], isGrayScale=config['is_grayscale'])

    logger.info(f"[ INFO ] Initialize {config['optimizer'].upper()} optimizer...")
    if config['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'nadam':
        # http://cs229.stanford.edu/proj2015/054_report.pdf
        optimizer = Nadam(learning_rate=config['learning_rate'])
    else:
        optimizer = SGD(learning_rate=config['learning_rate'])

    logger.info('[ INFO ] Initialize Trainer...')
    target = config['scale'] * config['patch_size']
    loss = Losses(targetShape=(target, target, 1))

    basename = os.path.basename(opt.cfg).split('.')[0]
    ckptDir = os.path.join(config['model_out'], f'ckpt_{basename}', opt.band)
    logDir = os.path.join(config['model_out'], f'logs_{basename}', opt.band)

    if config['loss'] == 'l1':
        type_loss = loss.shiftCompensatedL1Loss
    elif config['loss'] == 'sobel_l1_mix':
        type_loss = loss.shiftCompensatedL1EdgeLoss
    elif config['loss'] == 'l2':
        type_loss = loss.shiftCompensatedL2Loss
    elif config['loss'] == 'l1msssim':
        type_loss = loss.shiftCompensatedRevSSIM

    trainClass = ModelTrainer(model=model,
                              loss=type_loss,
                              metric=loss.shiftCompensatedcPSNR,
                              optimizer=optimizer,
                              ckptDir=ckptDir,
                              logDir=logDir)

    trainClass.fitTrainData(X_train, y, config['batch_size'], config['epochs'], valData,
                            saveBestOnly=False, initEpoch=0)

    logger.info(f'[ SUCCESS ] Model checkpoint can be found in {ckptDir}.')
    logger.info(f'[ SUCCESS ] Model logs can be found in {logDir}.')


def fusionNet(config):
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
        inputImgs, allImgMsk, ~allImgMsk.mask, test_size=config['split'], random_state=17)

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

    logger.info(f"[ INFO ] Initialize {config['optimizer'].upper()} optimizer...")
    if config['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'nadam':
        # http://cs229.stanford.edu/proj2015/054_report.pdf
        optimizer = Nadam(learning_rate=config['learning_rate'])
    else:
        optimizer = SGD(learning_rate=config['learning_rate'])

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

    trainClass.fitTrainData(X_train, y, config['batch_size'], config['epochs'], valData)

    logger.info(f'[ SUCCESS ] Model checkpoint can be found in {ckptDir}.')
    logger.info(f'[ SUCCESS ] Model logs can be found in {logDir}.')


if __name__ == '__main__':
    opt = parser()
    config = parseConfig(opt.cfg)
    if opt.modelType == 'patchNet':
        patchNet(config)
    else:
        fusionNet(config)
