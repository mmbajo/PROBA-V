from typing import List, Tuple, Dict
import argparse

from scipy.ndimage import fourier_shift, shift
from skimage.feature import register_translation, masked_register_translation
from skimage.transform import rescale
from skimage import io
from shutil import move
from tqdm import tqdm
from parseConfig import parseConfig
import torch
import random
import pandas as pd
import numpy as np
import glob
import os
import gc

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfg/p16t9c85r12.cfg', type=str)
    parser.add_argument('--band', default='NIR', type=str)
    opt = parser.parse_args()
    return opt


def main(config):
    rawDataDir = config['raw_data']
    cleanDataDir = config['preprocessing_out']
    band = opt.band
    arrayDir = os.path.join(cleanDataDir, 'arrayDir')
    trimmedArrayDir = os.path.join(cleanDataDir, 'trimmedArrayDir')
    patchesDir = os.path.join(cleanDataDir, 'patchesDir')
    trimmedPatchesDir = os.path.join(cleanDataDir, 'trimmedPatchesDir')
    augmentedPatchesDir = os.path.join(cleanDataDir, 'augmentedPatchesDir')

    # Check validity of directories
    if not os.path.exists(arrayDir):
        os.makedirs(arrayDir)
    if not os.path.exists(trimmedArrayDir):
        os.makedirs(trimmedArrayDir)
    if not os.path.exists(patchesDir):
        os.makedirs(patchesDir)
    if not os.path.exists(trimmedPatchesDir):
        os.makedirs(trimmedPatchesDir)
    if not os.path.exists(augmentedPatchesDir):
        os.makedirs(augmentedPatchesDir)

    # CHECKPOINT 1 - RAW DATA LOAD AND SAVE
    if 1 in config['ckpt']:
        # Train
        logging.info('Loading and dumping raw data...')
        loadAndSaveRawData(rawDataDir, arrayDir, 'NIR', isGrayScale=True, isTrainData=True)
        loadAndSaveRawData(rawDataDir, arrayDir, 'RED', isGrayScale=True, isTrainData=True)
        # Test
        loadAndSaveRawData(rawDataDir, arrayDir, 'NIR', isGrayScale=True, isTrainData=False)
        loadAndSaveRawData(rawDataDir, arrayDir, 'RED', isGrayScale=True, isTrainData=False)

    # CHECKPOINT 2 - IMAGE REGISTRATION AND CORRUPTED IMAGE SET REMOVAL
    if 2 in config['ckpt']:
        # Load dataset
        logging.info(f'Loading {band} dataset...')
        TRAIN, TEST = loadData(arrayDir, band)

        # Process the train dataset
        logging.info(f'Processing {band} train dataset...')
        allImgLR, allMskLR, allImgHR, allMskHR = TRAIN
        allImgMskLR = registerImages(allImgLR, allMskLR)  # np.ma.masked_array
        allImgMskHR = convertToMaskedArray(allImgHR, allMskHR)  # np.ma.masked_array
        trmImgMskLR, trmImgMskHR = removeCorruptedTrainImageSets(
            allImgMskLR, allImgMskHR, clarityThreshold=config['low_res_threshold'])
        trmImgMskLR = pickClearLRImgsPerImgSet(
            trmImgMskLR, numImgToPick=config['num_low_res_imgs'], clarityThreshold=config['low_res_threshold'])

        # Process the test dataset
        logging.info(f'Processing {band} test dataset...')
        allImgLRTest, allMskLRTest = TEST
        allImgMskLRTest = registerImages(allImgLRTest, allMskLRTest)  # np.ma.masked_array
        trmImgMskLRTest = removeCorruptedTestImageSets(allImgMskLRTest, clarityThreshold=config['low_res_threshold'])
        trmImgMskLRTest = pickClearLRImgsPerImgSet(
            trmImgMskLRTest, numImgToPick=config['num_low_res_imgs'], clarityThreshold=config['low_res_threshold'])

        logging.info(f'Saving {band} trimmed dataset...')
        if not os.path.exists(trimmedArrayDir):
            os.makedirs(trimmedArrayDir)
        trmImgMskLR.dump(os.path.join(trimmedArrayDir, f'TRAINimgLR_{band}.npy'))
        trmImgMskHR.dump(os.path.join(trimmedArrayDir, f'TRAINimgHR_{band}.npy'))
        trmImgMskLRTest.dump(os.path.join(trimmedArrayDir, f'TESTimgLR_{band}.npy'))

    # CHECKPOINT 3 - PATCH GENERATION
    if 3 in config['ckpt']:
        # Generate patches
        logging.info(f'Loading TEST {band} LR patch dataset...')
        trmImgMskLRTest = np.load(os.path.join(trimmedArrayDir, f'TESTimgLR_{band}.npy'), allow_pickle=True)
        logging.info(f'Generating TEST {band} LR Patches...')
        numImgSet, numImgPerImgSet, C, _, _ = trmImgMskLRTest.shape
        # padding with size of Loss Crop cropBorder
        if config['max_shift'] > 0:
            cropBorder = config['max_shift']//2
            paddings = [[0, 0], [0, 0], [0, 0], [cropBorder,
                                                 cropBorder], [cropBorder, cropBorder]]
            trmImgLRTest = np.pad(trmImgMskLRTest, paddings, 'reflect')
            trmMskLRTest = np.pad(trmImgMskLRTest.mask, paddings, 'reflect')
            trmImgMskLRTest = np.ma.masked_array(trmImgLRTest, mask=trmMskLRTest)

        patchesLR = generatePatches(
            trmImgMskLRTest, patchSize=config['patch_size'] + config['max_shift'], stride=config['patch_size'])
        patchesLR = patchesLR.reshape((numImgSet, -1, numImgPerImgSet, C,
                                       config['patch_size'] + config['max_shift'],
                                       config['patch_size'] + config['max_shift']))
        logging.info(f'Saving {band} LR Patches...')
        patchesLR.dump(os.path.join(patchesDir, f'TESTpatchesLR_{band}.npy'), protocol=4)
        del trmImgMskLRTest
        del patchesLR
        gc.collect()

        logging.info(f'Loading TRAIN {band} LR patch dataset...')
        trmImgMskLR = np.load(os.path.join(trimmedArrayDir, f'TRAINimgLR_{band}.npy'), allow_pickle=True)

        logging.info(f'Generating TRAIN {band} LR Patches...')
        numImgSet, numImgPerImgSet, C, H, W = trmImgMskLR.shape
        if config['max_shift'] > 0:
            cropBorder = config['max_shift']//2
            paddings = [[0, 0], [0, 0], [0, 0], [cropBorder,
                                                 cropBorder], [cropBorder, cropBorder]]
            trmImgLR = np.pad(trmImgMskLR, paddings, 'reflect')
            trmMskLR = np.pad(trmImgMskLR.mask, paddings, 'reflect')
            trmImgMskLR = np.ma.masked_array(trmImgLR, mask=trmMskLR)

        patchesLR = generatePatches(
            trmImgMskLR, patchSize=config['patch_size'] + config['max_shift'], stride=config['patch_stride'])
        patchesLR = patchesLR.reshape((numImgSet, -1, numImgPerImgSet, C,
                                       config['patch_size'] + config['max_shift'],
                                       config['patch_size'] + config['max_shift']))
        logging.info(f'Saving {band} LR Patches...')
        patchesLR.dump(os.path.join(patchesDir, f'TRAINpatchesLR_{band}.npy'), protocol=4)
        del trmImgMskLR
        del patchesLR
        gc.collect()

        logging.info(f'Loading TRAIN {band} HR patch dataset...')
        trmImgMskHR = np.load(os.path.join(trimmedArrayDir, f'TRAINimgHR_{band}.npy'), allow_pickle=True)
        logging.info(f'Generating TRAIN {band} HR Patches...')
        numImgSet, numImgPerImgSet, C, Hhr, Whr = trmImgMskHR.shape
        # Compute upsampleScale
        upsampleScale = Hhr // H
        patchesHR = generatePatches(trmImgMskHR, patchSize=config['patch_size'] *
                                    upsampleScale, stride=config['patch_size'] * upsampleScale)
        patchesHR = patchesHR.reshape((numImgSet, -1, numImgPerImgSet, C, config['patch_size'] *
                                       upsampleScale, config['patch_size'] * upsampleScale))
        logging.info(f'Saving {band} HR Patches...')
        patchesHR.dump(os.path.join(patchesDir, f'TRAINpatchesHR_{band}.npy'), protocol=4)
        del trmImgMskHR
        del patchesHR
        gc.collect()

    # CHECKPOINT 4 - CLEANING PATCHES
    if 4 in config['ckpt']:
        logging.info(f'Loading {band} test LR Patches...')
        patchesLRTest = np.load(os.path.join(patchesDir, f'TESTpatchesLR_{band}.npy'), allow_pickle=True)
        logging.info(f'Loading {band} train LR Patches...')
        patchesLR = np.load(os.path.join(patchesDir, f'TRAINpatchesLR_{band}.npy'), allow_pickle=True)
        logging.info(f'Loading {band} train HR Patches...')
        patchesHR = np.load(os.path.join(patchesDir, f'TRAINpatchesHR_{band}.npy'), allow_pickle=True)
        logging.info(f'Remove corrupted train {band} Patch sets...')
        trmPatchesLR, trmPatchesHR = removeCorruptedTrainPatchSets(
            patchesLR, patchesHR, clarityThreshold=config['high_res_threshold'])

        logging.info(f"Deleting {band} test LR patches that has below {config['low_res_threshold']} clarity...")
        trmPatchesLRTest = pickClearPatchesLR(patchesLRTest, clarityThreshold=config['low_res_threshold'])

        logging.info(f"Deleting {band} train LR patches that has below {config['low_res_threshold']} clarity...")
        trmPatchesLR = pickClearPatchesLR(trmPatchesLR, clarityThreshold=config['low_res_threshold'])

        logging.info(f"Deleting {band} train HR patches that has below {config['high_res_threshold']} clarity...")
        trmPatchesLR, trmPatchesHR = pickClearPatches(
            trmPatchesLR, trmPatchesHR, clarityThreshold=config['high_res_threshold'])

        # Reshape to [N, C, D, H, W] for PyTorch training
        logging.info(f'Reshaping {band} train patches...')
        trmPatchesLR = trmPatchesLR.transpose((0, 3, 4, 1, 2))  # shape is (numImgSet, H, W, numLRImg, C)
        trmPatchesHR = trmPatchesHR.transpose((0, 3, 4, 1, 2))
        trmPatchesHR = trmPatchesHR.squeeze(4)  # (numImgSet, H, W, C)

        logging.info(f'Saving {band} train patches...')
        trmPatchesLRTest.dump(os.path.join(trimmedPatchesDir, f'TESTpatchesLR_{band}.npy'), protocol=4)
        trmPatchesLR.dump(os.path.join(trimmedPatchesDir, f'TRAINpatchesLR_{band}.npy'), protocol=4)
        trmPatchesHR.dump(os.path.join(trimmedPatchesDir, f'TRAINpatchesHR_{band}.npy'), protocol=4)

    # CHECKPOINT 5 - AUGMENTING PATCHES
    if 5 in config['ckpt']:
        logging.info(f'Loading {band} train LR Patches...')
        augmentedPatchesLR = np.load(os.path.join(trimmedPatchesDir, f'TRAINpatchesLR_{band}.npy'), allow_pickle=True)
        logging.info(f'Augmenting by permuting {band} train HR Patches... Input: {augmentedPatchesLR.shape}')
        augmentedPatchesLR = augmentByShufflingLRImgs(augmentedPatchesLR, numPermute=config['num_low_res_permute'])
        if config['to_flip'] or config['to_flip']:
            logging.info(
                f'Augmenting by flipping/rotating {band} train LR Patches... Input: {augmentedPatchesLR.shape}')
            augmentedPatchesLR = augmentByFlipping(augmentedPatchesLR)
        logging.info(f'Saving {band} train LR Patches... Final shape: {augmentedPatchesLR.shape}')
        augmentedPatchesLR.dump(os.path.join(augmentedPatchesDir, f'TRAINpatchesLR_{band}.npy'), protocol=4)
        del augmentedPatchesLR
        gc.collect()

        logging.info(f'Loading {band} train HR Patches...')
        augmentedPatchesHR = np.load(os.path.join(trimmedPatchesDir, f'TRAINpatchesHR_{band}.npy'), allow_pickle=True)
        logging.info(f'Augmenting by permuting {band} train HR Patches... Input: {augmentedPatchesHR.shape}')
        augmentedPatchesHR = np.tile(augmentedPatchesHR, (config['num_low_res_permute'] + 1, 1, 1, 1))
        if config['to_flip'] or config['to_flip']:
            logging.info(
                f'Augmenting by flipping/rotating {band} train HR Patches... Input: {augmentedPatchesHR.shape}')
            augmentedPatchesHR = augmentByFlipping(augmentedPatchesHR)
        logging.info(f'Saving {band} train HR Patches... Final shape: {augmentedPatchesHR.shape}')
        augmentedPatchesHR.dump(os.path.join(augmentedPatchesDir, f'TRAINpatchesHR_{band}.npy'), protocol=4)
        del augmentedPatchesHR
        gc.collect()


def augmentByRICAP():
    pass


def augmentByShufflingLRImgs(patchLR: np.ma.masked_array, numPermute=9):
    # shape is (numImgSet, H, W, numLRImg, C)
    # (numImgSet, H, W, C)
    if numPermute == 0:
        return patchLR
    numImgSet, H, W, numLRImg, C = patchLR.shape
    cacheLR = [patchLR]
    for _ in range(numPermute):
        idx = np.random.permutation(np.arange(numLRImg))
        shuffled = patchLR[:, :, :, idx, :]
        cacheLR.append(shuffled)
    patchLR = np.concatenate(cacheLR)
    return patchLR


def augmentByFlipping(patches: np.ma.masked_array):
    img90 = np.rot90(patches, k=1, axes=(1, 2))
    img180 = np.rot90(patches, k=2, axes=(1, 2))
    img270 = np.rot90(patches, k=3, axes=(1, 2))
    imgFlipV = np.flip(patches, axis=(1))
    imgFlipH = np.flip(patches, axis=(2))
    imgFlipVH = np.flip(patches, axis=(1, 2))

    allImgMsk = np.concatenate((patches, img90, img180, img270, imgFlipV, imgFlipH, imgFlipVH))
    return allImgMsk


def pickClearPatchesLR(patchesLR: np.ma.masked_array,
                       clarityThreshold: float) -> List[np.ma.masked_array]:
    '''
    Input:
    patchesLR: np.ma.masked_array[numImgSet, numPatches, numLowResImg, C, H, W]
    clarityThreshold: float

    Output:
    cleanPatchesLR: np.ma.masked_array[numImgSet, numPatches, numLowResImg, C, H, W]

    '''
    desc = '[ INFO ] Cleaning train patches        '
    count = 0
    countNotReplacedAll = 0
    numImgSet, numPatches, numLowResImg, C, H, W = patchesLR.shape
    cache = []
    for imgSet in tqdm(patchesLR, desc=desc):
        cleanedImgSet, countNotGood, countNotReplaced = removeAndReplaceDirtyFrames(imgSet, clarityThreshold)
        cache.append(cleanedImgSet)
        count += countNotGood
        countNotReplacedAll += countNotReplaced
    trimmedPatchesLR = np.ma.array(cache)
    notGood = (count/(numImgSet*numPatches) * 100)
    notReplaced = countNotReplacedAll/count * 100
    if notGood > 50:
        print(f'[ WARNING ] {notGood:.2f}% of the patches did not pass the {clarityThreshold} threshold.')
        print(f'[ WARNING ] Among those patches, {notReplaced:.2f}% were not replaced!')
    else:
        print(f'[ INFO ] {notGood:.2f}% of the patches did not pass the {clarityThreshold} threshold.')
        print(f'[ INFO ] Among those patches, {notReplaced:.2f}% were not replaced!')
    return trimmedPatchesLR


def removeAndReplaceDirtyFrames(imgSet: np.ma.masked_array, clarityThreshold: float) -> np.ma.masked_array:
    '''
    Input:
    imgSet: np.ma.masked_array[numPatches, numLowResImg, C, H, W]
    clarityThreshold: float

    Output:
    cleanPatchesLR: np.ma.masked_array[numPatches, numLowResImg, C, H, W]
    '''
    cache = []
    numPatches, numLowResImg, C, H, W = imgSet.shape
    count = 0
    countNotReplaced = 0
    for patch in imgSet:
        # [numLowResImg, C, H, W]
        booleanMask = np.array([np.count_nonzero(lr.mask)/(H * W) < (1-clarityThreshold) for lr in patch])
        trimmedPatch = patch[booleanMask]
        if len(trimmedPatch) == 0:
            cache.append(patch)
            count += numLowResImg
            countNotReplaced += numLowResImg
            continue
        endPatch = patch[booleanMask]
        count += (numLowResImg - len(endPatch))
        while len(endPatch) < numLowResImg:
            endPatch = np.ma.concatenate((endPatch, trimmedPatch))

        endPatch = endPatch[:numLowResImg]
        cache.append(endPatch)
    return np.ma.array(cache), count, countNotReplaced


def pickClearPatches(patchesLR: np.ma.masked_array,
                     patchesHR: np.ma.masked_array,
                     clarityThreshold: float) -> List[np.ma.masked_array]:
    '''
    Input:
    patchesLR: np.ma.masked_array[numImgSet, numPatches, numLowResImg, C, H, W]
    patchesHR: np.ma.masked_array[numImgSet, numPatches, 1, C, H, W]
    clarityThreshold: float

    Output:
    cleanPatchesLR: np.ma.masked_array[numImgSet*newNumPatches, numLowResImg, C, H, W]
    cleanPatchesHR: np.ma.masked_array[numImgSet*newNumPatches, 1, C, H, W]
                        where newNumPatches <= numPatches
    '''
    desc = '[ INFO ] Cleaning train patches        '
    numImgSet, numPatches, numLowResImg, C, HLR, WLR = patchesLR.shape
    reshapeLR = patchesLR.reshape((-1, numLowResImg, C, HLR, WLR))
    _, _, numHighResImg, C, HHR, WHR = patchesHR.shape
    reshapeHR = patchesHR.reshape((-1, numHighResImg, C, HHR, WHR))
    booleanMask = np.array([isPatchNotCorrupted(patch, clarityThreshold)
                            for patch in tqdm(reshapeHR, desc=desc)])
    trimmedPatchesLR = reshapeLR[booleanMask]
    trimmedPathcesHR = reshapeHR[booleanMask]
    return (trimmedPatchesLR, trimmedPathcesHR)


def pickClearPatchesV2(patchesLR: np.ma.masked_array,
                       patchesHR: np.ma.masked_array,
                       clarityThreshold: float) -> np.array:
    '''
    Input:
    patchesLR: np.ma.masked_array[numImgSet, numPatches, numLowResImg, C, H, W]
    patchesHR: np.ma.masked_array[numImgSet, numPatches, 1, C, H, W]
    clarityThreshold: float

    Output:
    cleanPatchesLR: np.ma.masked_array[numImgSet, newNumPatches, numLowResImg, C, H, W]
    cleanPatchesHR: np.ma.masked_array[numImgSet, newNumPatches, 1, C, H, W]
                        where newNumPatches <= numPatches
    '''
    desc = '[ INFO ] Cleaning train patches        '
    trmPatchesLR, trmPatchesHR = [], []
    for patchSetLR, patchSetHR in tqdm(zip(patchesLR, patchesHR), desc=desc, total=len(patchesLR)):
        trmPatchSetLR, trmPatchSetHR = explorePatchSet(patchSetLR, patchSetHR, clarityThreshold)
        trmPatchesLR.append(trmPatchSetLR)
        trmPatchesHR.append(trmPatchSetHR)
    return (np.array(trmPatchesLR), np.array(trmPatchesHR))


def explorePatchSet(patchSetLR: np.ma.masked_array,
                    patchSetHR: np.ma.masked_array,
                    clarityThreshold: float) -> List[np.ma.array]:
    '''
    Explores a patch set and removes patches that do not have enough clarity.

    Input:
    patchSetLR: np.ma.masked_array[numPatches, numLowResImg, C, H, W],
    patchSetHR: np.ma.masked_array[numPatches, 1, C, H, W],
    clarityThreshold: float
    '''
    booleanMask = np.array([isPatchNotCorrupted(patch, clarityThreshold)
                            for patch in patchSetHR])
    trmPatchSetLR = patchSetLR[booleanMask]
    trmPatchSetHR = patchSetHR[booleanMask]
    return trmPatchSetLR, trmPatchSetHR


def isPatchNotCorrupted(patch: np.ma.masked_array, clarityThreshold: float) -> bool:
    '''
    Determine if an HR patch passes the threshold

    Input:
    patchSet: np.ma.masked_array[1, C, H, W]
    clarityThreshold: float

    Output:
    boolean that answers the question is Patch good enough?
    '''
    isPatchClearEnough = np.count_nonzero(patch.mask)/(patch.shape[2] * patch.shape[3]) < (1-clarityThreshold)
    return isPatchClearEnough


def removeCorruptedTrainPatchSets(patchesLR: np.ma.masked_array,
                                  patchesHR: np.ma.masked_array,
                                  clarityThreshold: float) -> List[np.ma.masked_array]:
    '''
    Input:
    patchesLR: np.ma.masked_array[numImgSet, numPatches, numLowResImg, C, H, W]
    patchesHR: np.ma.masked_array[numImgSet, numPatches, 1, C, H, W]
    clarityThreshold: float

    Output:
    cleanPatchesLR: np.ma.masked_array[numImgSet, newNumPatches, numLowResImg, C, H, W]
    cleanPatchesHR: np.ma.masked_array[numImgSet, newNumPatches, 1, C, H, W]
                        where newNumPatches <= numPatches
    '''
    desc = '[ INFO ] Removing corrupted train sets '
    booleanMask = np.array([isPatchSetNotCorrupted(patchSet, clarityThreshold)
                            for patchSet in tqdm(patchesHR, desc=desc)])
    trimmedPatchesLR = patchesLR[booleanMask]
    trimmedPathcesHR = patchesHR[booleanMask]
    return (trimmedPatchesLR, trimmedPathcesHR)


def removeCorruptedTestPatchSets(patchesLR: np.ma.masked_array,
                                 clarityThreshold: float) -> np.ma.masked_array:
    '''
    Input:
    patchesLR: np.ma.masked_array[numImgSet, numPatches, numLowResImg, C, H, W]
    clarityThreshold: float

    Output:
    cleanPatchesLR: np.ma.masked_array[numImgSet, newNumPatches, numLowResImg, C, H, W]
                        where newNumPatches <= numPatches
    '''
    desc = '[ INFO ] Removing corrupted test sets  '
    booleanMask = np.array([isPatchSetNotCorrupted(patchSet, clarityThreshold)
                            for patchSet in tqdm(patchesLR, desc=desc)])
    trimmedPatchesLR = patchesLR[booleanMask]
    return trimmedPatchesLR


def isPatchSetNotCorrupted(patchSet: np.ma.masked_array, clarityThreshold: float) -> bool:
    '''
    Determine if all the LR images are not clear enough.
    Return False if ALL LR image clarity is below threshold.

    Input:
    patchSet: np.ma.masked_array[numPatches, numLowResImg, C, H, W]
    clarityThreshold: float

    Output:
    boolean that answers the question is PatchSet not Corrupted?
    '''
    isPatchClearEnough = np.array([np.count_nonzero(patch.mask)/(patch.shape[-1]*patch.shape[-2]) < (1-clarityThreshold)
                                   for patch in patchSet])
    return np.sum(isPatchClearEnough) != 0


def generatePatches(imgSets: np.ma.masked_array, patchSize: int, stride: int) -> np.ma.masked_array:
    '''
    Input:
    images: np.ma.masked_array[numImgSet, numImgPerImgSet, channels, height, width]
    patchSize: int
    stride: int

    Output:
    np.ma.masked_array[numImgSet, numImgPerImgSet * numPatches, channels, patchSize, patchSize]
    '''
    desc = f'[ INFO ] Generating patches (k={patchSize}, s={stride})'
    if imgSets.dtype != 'float32':
        imgSets = imgSets.astype(np.float32)
    return np.ma.array([generatePatchesPerImgSet(imgSet, patchSize, stride) for imgSet in tqdm(imgSets, desc=desc)])


def generatePatchesPerImgSet(images: np.ma.masked_array, patchSize: int, stride: int) -> np.ma.masked_array:
    '''
    Generate patches of images systematically.

    Input:
    images: np.ma.masked_array[numImgPerImgSet, channels, height, width]
    patchSize: int
    stride: int

    Output:
    np.ma.masked_array[numImgPerImgSet * numPatches, channels, patchSize, patchSize]
    '''
    tensorImg = torch.tensor(images)
    tensorMsk = torch.tensor(images.mask)

    numMskPerImgSet, channels, height, width = images.shape

    patchesImg = tensorImg.unfold(0, numMskPerImgSet, numMskPerImgSet).unfold(
        1, channels, channels).unfold(2, patchSize, stride).unfold(3, patchSize, stride)
    patchesImg = patchesImg.reshape(-1, channels, patchSize, patchSize)  # [numImgPerImgSet * numPatches, C, H, W]
    patchesImg = patchesImg.numpy()

    patchesMsk = tensorMsk.unfold(0, numMskPerImgSet, numMskPerImgSet).unfold(
        2, patchSize, stride).unfold(3, patchSize, stride)
    patchesMsk = patchesMsk.reshape(-1, channels, patchSize, patchSize)
    patchesMsk = patchesMsk.numpy()

    return np.ma.masked_array(patchesImg, mask=patchesMsk)


def registerImages(allImgLR: np.ndarray, allMskLR: np.ndarray) -> np.ma.masked_array:
    '''
    For each imgset, align all its imgs into one coordinate system.
    The reference image will be the clearest one. (ie the one withe highest QM accumalitive sum)

    Input:
    allImgLR: np.ndarray[numImgSet, numImgPerImgSet, channel, height, width]
    allMskLR: np.ndarray[numImgSet, numMskPerImgSet, channel, height, width]

    Output:
    output: np.ma.masked_array with the same dimension
    '''
    desc = '[ INFO ] Registering LR images         '
    return np.ma.array([registerImagesInSet(allImgLR[i], allMskLR[i])
                        for i in tqdm(range(allImgLR.shape[0]), desc=desc)])


def registerImagesInSet(imgLR: np.ndarray, mskLR: np.ndarray) -> np.ma.masked_array:
    '''
    Takes in an imgset LR masks and images.
    Sorts it and picks a reference, then register.

    Input:
    imgLR: np.ndarray[numImgPerImgSet, channel, height, width]
    mskLR: np.ndarray[numMskPerImgSet, channel, height, width]

    Output
    regImgMskLR: np.ma.masked_array[numMskPerImgSet, channel, height, width]
                    This array has a property mask where in if used, returns a boolean array
                    with the same dimension as the data.
    https://docs.scipy.org/doc/numpy-1.15.0/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray.data
    '''
    sortedIdx = np.argsort([-np.count_nonzero(msk) for msk in mskLR])
    clearestToDirtiestImg = imgLR[sortedIdx]
    clearestToDirtiestMsk = mskLR[sortedIdx]
    referImg = clearestToDirtiestImg[0]
    for i, (img, msk) in enumerate(zip(clearestToDirtiestImg, clearestToDirtiestMsk)):
        if i == 0:
            regImgMskLR = np.expand_dims(np.ma.masked_array(img, mask=~msk), axis=0)
        else:
            regImg, regMsk = registerFrame(img, msk.astype(bool), referImg)
            mskdArray = np.expand_dims(np.ma.masked_array(regImg, mask=~(regMsk > 0)), axis=0)
            regImgMskLR = np.ma.concatenate((regImgMskLR, mskdArray))
    return regImgMskLR


def registerFrame(img: np.ndarray, msk: np.ndarray, referenceImg: np.ndarray, tech='freq') -> Tuple[np.ndarray, np.ndarray]:
    '''
    Input:
    img: np.ndarray[channel, height, width]
    msk: np.ndarray[channel, height, width]
    referenceImg: np.ndarray[channel, height, width]

    Output:
    Tuple(regImg, regMsk)
    regImg: np.ndarray[channel, height, width]
    regMsk: np.ndarray[channel, height, width]
    '''
    if tech == 'time':
        shiftArray = masked_register_translation(referenceImg, img, msk)
        regImg = shift(img, shiftArray, mode='reflect')
        regMsk = shift(msk, shiftArray, mode='constant', cval=0)
    if tech == 'freq':
        shiftArray, _, _ = register_translation(referenceImg, img)
        regImg = fourier_shift(np.fft.fftn(img), shiftArray)
        regImg = np.fft.ifftn(regImg)
        regImg = regImg.real

        regMsk = fourier_shift(np.fft.fftn(msk), shiftArray)
        regMsk = np.fft.ifftn(regMsk)
        regMsk = regMsk.real
    return (regImg, regMsk)


def convertToMaskedArray(imgSets: np.ndarray, mskSets: np.ndarray) -> np.ma.masked_array:
    '''
    Convert Image and Mask array pair to a masked array.
    Especially made for HR images.

    Input:
    imgSets: np.ndarray[numImgSet, numImgPerImgSet, channel, height, width]
    mskSets: np.ndarray[numImgSet, numImgPerImgSet, channel, height, width]

    Output:
    imgMskSets: np.ma.masked_array[numImgSet, numImgPerImgSet, channel, height, width]
    '''
    imgSets = np.squeeze(imgSets, axis=1)  # [numImgSet, channel, height, width]
    mskSets = np.squeeze(mskSets, axis=1)  # [numImgSet, channel, height, width]
    imgMskSets = np.ma.array([np.ma.masked_array(img, mask=~msk)
                              for img, msk in zip(imgSets, mskSets)])  # [numImgSet, channel, height, width]
    imgMskSets = np.expand_dims(imgMskSets, axis=1)  # [numImgSet, 1, channel, height, width]
    return imgMskSets


def removeCorruptedTrainImageSets(imgMskLR: np.ma.masked_array, imgMskHR: np.ma.masked_array,
                                  clarityThreshold: float) -> Tuple[np.ma.masked_array, np.ma.masked_array]:
    '''
    Remove imageset if ALL its LR frames is less than the given clarity threshold.

    Input:
    imgMskLR: np.ma.masked_array[numImgSet, numImgPerImgSet, channel, height, width]
    imgMskHR: np.ma.masked_array[numImgSet,               1, channel, height, width]
    clarityThreshold: float

    Output:
    trimmedImgMskLR: np.ma.masked_array[newNumImgSet, numImgPerImgSet, channel, height, width]
    trimmedImgMskHR: np.ma.masked_array[newNumImgSet,               1, channel, height, width]
                    where newNumImgSet <= numImgSet
    '''
    desc = '[ INFO ] Removing corrupted ImageSets  '
    booleanMask = np.array([isImageSetNotCorrupted(imgSet, clarityThreshold) for imgSet in tqdm(imgMskLR, desc=desc)])
    trimmedImgMskLR = imgMskLR[booleanMask]
    trimmedImgMskHR = imgMskHR[booleanMask]
    return (trimmedImgMskLR, trimmedImgMskHR)


def removeCorruptedTestImageSets(imgMskLR: np.ma.masked_array,
                                 clarityThreshold: float) -> np.ma.masked_array:
    '''
    Remove imageset if ALL its LR frames is less than the given clarity threshold.

    Input:
    imgMskLR: np.ma.masked_array[numImgSet, numImgPerImgSet, channel, height, width]
    clarityThreshold: float

    Output:
    trimmedImgMskLR: np.ma.masked_array[newNumImgSet, numImgPerImgSet, channel, height, width]
                    where newNumImgSet <= numImgSet
    '''
    desc = '[ INFO ] Removing corrupted ImageSets  '
    booleanMask = np.array([isImageSetNotCorrupted(imgSet, clarityThreshold)
                            for imgSet in tqdm(imgMskLR, desc=desc)])
    trimmedImgMskLR = imgMskLR[booleanMask]
    return trimmedImgMskLR


def isImageSetNotCorrupted(imgSet: np.ma.masked_array, clarityThreshold: float) -> bool:
    '''
    Determine if all the LR images are not clear enough.
    Return False if ALL LR image clarity is below threshold.

    Input:
    imgSet: np.ma.masked_array[numImgPerImgSet, channel, height, width]
    clarityThreshold: float

    Output:
    boolean that answers the question is ImageSet not Corrupted?
    '''
    # totalPixels = imgSet.shape[2] * imgSet.shape[3]  # width * height
    isImageClearEnough = np.array([np.count_nonzero(img.mask)/(img.shape[1] * img.shape[2]) < (1-clarityThreshold)
                                   for img in imgSet])
    return np.sum(isImageClearEnough) != 0


def pickClearLRImgsPerImgSet(imgMskLR: np.ma.masked_array,
                             numImgToPick: int, clarityThreshold: float) -> np.ma.masked_array:
    '''
    Pick clearest frames per ImgSet.
    Before picking, we remove all frames that don't satisfy the clarity threshold.
    After removing the said frames, in the event that the remaining LR frames is less than
    the number of img to pick, we randomly pick among the clear frames to satisfy number of frames.
    (This might be a form of regularization...)

    Input:
    imgMskLR: np.ma.masked_array[newNumImgSet, numImgPerImgSet, channel, height, width]
    numImgToPick: int

    Output:
    trimmedImgMskLR: np.ma.masked_array[newNumImgSet, numImgToPick, channel, height, width]
                        where numImgToPick <= numImgPerImgSet
    '''
    desc = f'[ INFO ] Picking top {numImgToPick} clearest images '
    cache = []
    count = 0
    numImgPerImgSet, C, H, W = imgMskLR[0].shape
    for imgMsk in tqdm(imgMskLR, desc=desc):
        clearData, countDuplicates = pickClearImg(filterImgMskSet(imgMsk, clarityThreshold), numImgToPick=numImgToPick)
        cache.append(clearData)
        count += countDuplicates
    duplicates = (count/(len(imgMskLR)*numImgPerImgSet))*100
    print(f'[ INFO ] Among the all the LR images, {duplicates:.7f}% are duplicates of high quality frames.')
    return np.ma.array(cache)


def pickClearImg(imgMsk: np.ma.masked_array, numImgToPick: int) -> np.ma.masked_array:
    '''
    Pick clearest low resolution images!

    Input:
    imgMsk: np.ma.masked_array[numImgPerImgSet, channel, height, width]
    numImgToPick: int

    Ouput:
    trimmedImgMsk: np.ma.masked_array[newNumImgPerImgSet, channel, height, width]
                    where newNumImgPerImgSet <= numImgPerImgSet might not hold.
    '''
    sortedIndices = np.argsort(np.sum(imgMsk.mask, axis=(1, 2, 3)))
    sortedImgMskArray = imgMsk[sortedIndices]
    count = 0
    if numImgToPick < len(imgMsk):
        trimmedImgMsk = sortedImgMskArray[:numImgToPick]
    else:
        trimmedImgMsk = np.copy(sortedImgMskArray)
        count += (numImgToPick - len(trimmedImgMsk))
        while len(trimmedImgMsk) < numImgToPick:
            shuffledIndices = np.random.choice(sortedIndices, size=len(sortedIndices), replace=False)
            toAppend = imgMsk[shuffledIndices]
            trimmedImgMsk = np.ma.concatenate((trimmedImgMsk, toAppend))
        trimmedImgMsk = trimmedImgMsk[:numImgToPick]
    return trimmedImgMsk, count


def filterImgMskSet(imgSet: np.ma.masked_array, clarityThreshold: float) -> np.ma.masked_array:
    '''
    This function is the same as isImageSetNotCorrupted.
    except that the out put is a masked version of its array input.

    Input:
    imgSet: np.ma.masked_array[numImgPerImgSet, channel, height, width]
    clarityThreshold: float

    Output:
    filteredImgSet: np.ma.masked_array[newNumImgPerImgSet, channel, height, width]
                        where newNumImgPerImgSet <= numImgPerImgSet
    '''
    # totalPixels = imgSet.shape[2] * imgSet.shape[3]  # width * height
    isImageClearEnough = np.array([np.count_nonzero(img.mask)/(img.shape[1] * img.shape[2]) < (1-clarityThreshold)
                                   for img in imgSet])  # boolean mask
    filteredImgSet = imgSet[isImageClearEnough]
    return filteredImgSet


def loadData(arrayDir: str, band: str):
    '''
    Input:
    arrayDir: str -> the path folder for which you saved .npy files
    band: str -> 'NIR' or 'RED'
    isTrainData: bool -> set to true if dealing with the train dataset

    Output:
    List[Tuple(train data), Tuple(test data)]
    '''
    # Check input dir validity
    if not os.path.exists(arrayDir):
        raise Exception("[ ERROR ] Folder path does not exists...")
    if not os.listdir(arrayDir):
        raise Exception("[ ERROR ] No files in the provided directory...")

    TRAINimgLR = np.load(os.path.join(arrayDir, f'TRAINimgLR_{band}.npy'), allow_pickle=True)
    TRAINimgHR = np.load(os.path.join(arrayDir, f'TRAINimgHR_{band}.npy'), allow_pickle=True)
    TRAINmskLR = np.load(os.path.join(arrayDir, f'TRAINmskLR_{band}.npy'), allow_pickle=True)
    TRAINmskHR = np.load(os.path.join(arrayDir, f'TRAINmskHR_{band}.npy'), allow_pickle=True)

    TESTimgLR = np.load(os.path.join(arrayDir, f'TESTimgLR_{band}.npy'), allow_pickle=True)
    TESTmskLR = np.load(os.path.join(arrayDir, f'TESTmskLR_{band}.npy'), allow_pickle=True)

    TRAIN = (TRAINimgLR, TRAINmskLR, TRAINimgHR, TRAINmskHR)
    TEST = (TESTimgLR, TESTmskLR)

    return TRAIN, TEST


def loadAndSaveRawData(rawDataDir: str, arrayDir: str, band: str, isGrayScale=True, isTrainData=True):
    '''
    This function loads every imageset and dumps it into one giant array.
    We do this because of memory constraints...
    If you have about 64 GB of ram and about 72~128GB of swap space,
    you might opt not using this function.

    Input:
    rawDataDir: str -> downloaded raw data directory
    arrayDir: str -> the directory to which you will dump the numpy array
    band: str -> 'NIR' or 'RED'
    isGrayScale: bool -> Set to true if you are dealing with grayscale image

    Output:
    Array file with dimensions
    [numImgSet, numImgPerImgSet, channel, height, width]
    '''
    # Check if arrayDir exist, build if not.
    if not os.path.exists(arrayDir):
        os.makedirs(arrayDir)

    # Is train data?
    key = 'TRAIN' if isTrainData else 'TEST'

    # Get directories (OS agnostic)
    trainDir = os.path.join(rawDataDir, key.lower(), band)
    dirList = sorted(glob.glob(os.path.join(trainDir, 'imgset*')))

    # Load all low resolution images in a massive array and dump!
    # The resulting numpy array has dimensions [numImgSet, numLowResImgPerImgSet, channel, height, width]
    descForImgLR = '[ INFO ] Loading LR images and dumping '
    imgLR = np.array([np.array([io.imread(fName).transpose((2, 0, 1)) if not isGrayScale
                                else np.expand_dims(io.imread(fName), axis=0)
                                for fName in sorted(glob.glob(os.path.join(dirName, 'LR*.png')))])
                      for dirName in tqdm(dirList, desc=descForImgLR)])

    imgLR.dump(os.path.join(arrayDir, f'{key}imgLR_{band}.npy'))

    # Load all low resolution masks in a massive array and dump!
    # The resulting numpy array has dimensions [numImgSet, numLowResMaskPerImgSet, channel, height, width]
    descForMaskLR = '[ INFO ] Loading LR masks and dumping  '
    mskLR = np.array([np.array([io.imread(fName).transpose((2, 0, 1)) if not isGrayScale
                                else np.expand_dims(io.imread(fName), axis=0)
                                for fName in sorted(glob.glob(os.path.join(dirName, 'QM*.png')))])
                      for dirName in tqdm(dirList, desc=descForMaskLR)])

    mskLR.dump(os.path.join(arrayDir, f'{key}mskLR_{band}.npy'))

    if isTrainData:
        # Load all high resolution images in a massive array and dump!
        # The resulting numpy array has dimensions [numImgSet, 1, channel, height, width]
        descForImgHR = '[ INFO ] Loading HR images and dumping '
        imgHR = np.array([io.imread(os.path.join(dirName, 'HR.png')).transpose((2, 0, 1)) if not isGrayScale
                          else np.expand_dims(io.imread(os.path.join(dirName, 'HR.png')), axis=0)
                          for dirName in tqdm(dirList, desc=descForImgHR)])
        # For count of HR pics which is 1.
        imgHR = np.expand_dims(imgHR, axis=1)
        imgHR.dump(os.path.join(arrayDir, f'{key}imgHR_{band}.npy'))

        # Load all high resolution images in a massive array and dump!
        # The resulting numpy array has dimensions [numImgSet, 1, channel, height, width]
        descForMaskHR = '[ INFO ] Loading HR masks and dumping  '
        mskHR = np.array([io.imread(os.path.join(dirName, 'SM.png')).transpose((2, 0, 1)) if not isGrayScale
                          else np.expand_dims(io.imread(os.path.join(dirName, 'SM.png')), axis=0)
                          for dirName in tqdm(dirList, desc=descForMaskHR)])
        # For count of HR pics which is 1.
        mskHR = np.expand_dims(mskHR, axis=1)
        mskHR.dump(os.path.join(arrayDir, f'{key}mskHR_{band}.npy'))


if __name__ == '__main__':
    opt = parser()
    config = parseConfig(opt.cfg)
    main(config)
