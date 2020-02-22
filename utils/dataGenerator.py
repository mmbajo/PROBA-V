from typing import List, Tuple, Dict
import argparse

from scipy.ndimage import fourier_shift, shift
from skimage.feature import register_translation, masked_register_translation
from skimage.transform import rescale
from skimage import io
from shutil import move
from tqdm import tqdm
import torch
import random
import pandas as pd
import numpy as np
import glob
import os

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


DEBUG = 0
# Set the data data directory
# Download data at https://kelvins.esa.int/proba-v-super-resolution/data/
DATA_BANK_DIRECTORY = '/home/mark/DataBank/probav_data/'
DATA_BANK_DIRECTORY_PREPROCESSING_CHKPT = '/home/mark/DataBank/PROBA-V-CHKPT'


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--band', default='NIR', type=str)
    parser.add_argument('--dir', type=str, default=DATA_BANK_DIRECTORY)
    parser.add_argument('--chkptdir', default='/home/mark/DataBank/PROBA-V-CHKPT', type=str)
    parser.add_argument('--split', type=float, default=0.7)
    parser.add_argument('--numTopClearest', type=int, default=9)
    parser.add_argument('--patchSizeLR', type=int, default=32)
    parser.add_argument('--patchStrideLR', type=int, default=2)
    parser.add_argument('--ckpt', type=int, nargs='+', default=[1, 2, 3, 4])
    opt = parser.parse_args()
    return opt


def main():
    rawDataDir = opt.dir
    cleanDataDir = opt.chkptdir
    band = opt.band
    arrayDir = os.path.join(cleanDataDir, 'arrayDir')
    trimmedArrayDir = os.path.join(cleanDataDir, 'trimmedArrayDir')
    patchesDir = os.path.join(cleanDataDir, 'patchesDir')

    # Check validity of directories
    if not os.path.exists(arrayDir):
        os.makedirs(arrayDir)
    if not os.path.exists(trimmedArrayDir):
        os.makedirs(trimmedArrayDir)
    if not os.path.exists(patchesDir):
        os.makedirs(patchesDir)

    # CHECKPOINT 1 - RAW DATA LOAD AND SAVE
    if 1 in opt.ckpt:
        # Train
        logging.info('Loading and dumping raw data...')
        loadAndSaveRawData(rawDataDir, arrayDir, 'NIR', isGrayScale=True, isTrainData=True)
        loadAndSaveRawData(rawDataDir, arrayDir, 'RED', isGrayScale=True, isTrainData=True)
        # Test
        loadAndSaveRawData(rawDataDir, arrayDir, 'NIR', isGrayScale=True, isTrainData=False)
        loadAndSaveRawData(rawDataDir, arrayDir, 'RED', isGrayScale=True, isTrainData=False)

    # CHECKPOINT 2 - IMAGE REGISTRATION AND CORRUPTED IMAGE SET REMOVAL
    if 2 in opt.ckpt:
        # Load dataset
        logging.info(f'Loading {band} dataset...')
        TRAIN, TEST = loadData(arrayDir, band)

        # Process the train dataset
        logging.info(f'Processing {band} train dataset...')
        allImgLR, allMskLR, allImgHR, allMskHR = TRAIN
        allImgMskLR = registerImages(allImgLR, allMskLR)  # np.ma.masked_array
        allImgMskHR = convertToMaskedArray(allImgHR, allMskHR)  # np.ma.masked_array
        trmImgMskLR, trmImgMskHR = removeCorruptedTrainImageSets(allImgMskLR, allImgMskHR, clarityThreshold=0.7)
        trmImgMskLR = pickClearLRImgsPerImgSet(trmImgMskLR, numImgToPick=opt.numTopClearest, clarityThreshold=0.7)

        # Process the test dataset
        logging.info(f'Processing {band} test dataset...')
        allImgLRTest, allMskLRTest = TEST
        allImgMskLRTest = registerImages(allImgLRTest, allMskLRTest)  # np.ma.masked_array
        trmImgMskLRTest = removeCorruptedTestImageSets(allImgMskLRTest, clarityThreshold=0.7)
        trmImgMskLRTest = pickClearLRImgsPerImgSet(
            trmImgMskLRTest, numImgToPick=opt.numTopClearest, clarityThreshold=0.7)

        logging.info(f'Saving {band} trimmed dataset...')
        if not os.path.exists(trimmedArrayDir):
            os.makedirs(trimmedArrayDir)
        trmImgMskLR.dump(os.path.join(trimmedArrayDir, f'TRAINimgLR_{band}.npy'))
        trmImgMskHR.dump(os.path.join(trimmedArrayDir, f'TRAINimgHR_{band}.npy'))
        trmImgMskLRTest.dump(os.path.join(trimmedArrayDir, f'TESTimgLR_{band}.npy'))

    # CHECKPOINT 3 - PATCH GENERATION
    if 3 in opt.ckpt:
        logging.info(f'Loading {band} patch dataset...')
        trmImgMskLR = np.load(os.path.join(trimmedArrayDir, f'TRAINimgLR_{band}.npy'), allow_pickle=True)
        trmImgMskHR = np.load(os.path.join(trimmedArrayDir, f'TRAINimgHR_{band}.npy'), allow_pickle=True)
        trmImgMskLRTest = np.load(os.path.join(trimmedArrayDir, f'TESTimgLR_{band}.npy'), allow_pickle=True)

        # Compute upsampleScale
        upsampleScale = trmImgMskHR.shape[-1] // trmImgMskLR.shape[-1]

        # Generate patches
        logging.info(f'Generating {band} Patches...')

        numImgSet, numImgPerImgSet, C, H, W = trmImgMskLR.shape
        patchesLR = generatePatches(trmImgMskLR, patchSize=opt.patchSizeLR, stride=opt.patchStrideLR)
        patchesLR = patchesLR.reshape((numImgSet, -1, numImgPerImgSet, C, H, W))
        patchesLR.dump(os.path.join(patchesDir, f'TRAINpatchesLR_{band}.npy'), protocol=4)

        numImgSet, numImgPerImgSet, C, H, W = trmImgMskHR.shape
        patchesHR = generatePatches(trmImgMskHR, patchSize=opt.patchSizeLR *
                                    upsampleScale, stride=opt.patchStrideLR * upsampleScale)
        patchesHR = patchesHR.reshape((numImgSet, -1, numImgPerImgSet, C, H, W))
        patchesHR.dump(os.path.join(patchesDir, f'TRAINpatchesHR_{band}.npy'), protocol=4)

    # CHECKPOINT 4 - CLEANING PATCHES
    if 4 in opt.ckpt:
        pass


def augmentByRICAP():
    pass


def augmentByShuffling():
    pass


def shufflePatchSetAndAdd():
    pass


def generatePatches(imgSets: np.ma.masked_array, patchSize: int, stride: int) -> np.ma.masked_array:
    '''
    Input:
    images: np.ma.masked_array[numImgSet, numImgPerImgSet, channels, height, width]
    patchSize: int
    stride: int

    Output:
    np.ma.masked_array[numImgSet, numImgPerImgSet * numPatches, channels, patchSize, patchSize]
    '''
    #       con'[ INFO ] Loading LR masks and dumping  '
    desc = f'[ INFO ] Generating patches (k={patchSize}, s={stride})'
    if imgSets.dtype != 'float32':
        imgSets = imgSets.astype(np.float32)
    return np.array([generatePatchesPerImgSet(imgSet, patchSize, stride) for imgSet in tqdm(imgSets, desc=desc)])


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

    _, channels, height, width = images.shape

    patchesImg = tensorImg.unfold(2, patchSize, stride).unfold(3, patchSize, stride)
    patchesImg = patchesImg.reshape(-1, channels, patchSize, patchSize)  # [numImgPerImgSet * numPatches, C, H, W]
    patchesImg = patchesImg.numpy()

    patchesMsk = tensorMsk.unfold(2, patchSize, stride).unfold(3, patchSize, stride)
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
    #      '[ INFO ] Loading LR masks and dumping  '
    desc = '[ INFO ] Registering LR images         '
    return np.array([registerImagesInSet(allImgLR[i], allMskLR[i]) for i in tqdm(range(allImgLR.shape[0]), desc=desc)])


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
    referImg = imgLR[np.argmax([np.count_nonzero(msk) for msk in mskLR])]
    for i, (img, msk) in enumerate(zip(imgLR, mskLR)):
        regImg, regMsk = registerFrame(img, msk.astype(bool), referImg)
        mskdArray = np.expand_dims(np.ma.masked_array(regImg, mask=regMsk), axis=0)
        if i == 0:
            regImgMskLR = mskdArray
        else:
            regImgMskLR = np.ma.concatenate((regImgMskLR, mskdArray))
    return regImgMskLR


def registerFrame(img: np.ndarray, msk: np.ndarray, referenceImg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    shiftArray = masked_register_translation(referenceImg, img, msk)
    regImg = shift(img, shiftArray, mode='reflect')
    regMsk = shift(msk, shiftArray, mode='constant', cval=0)
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
    imgMskSets = np.ma.array([np.ma.masked_array(img, mask=msk)
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
    #      '[ INFO ] Loading LR masks and dumping  '
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
    booleanMask = np.array([isImageSetNotCorrupted(imgSet, clarityThreshold) for imgSet in tqdm(imgMskLR, desc=desc)])
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
    isImageClearEnough = np.array([np.count_nonzero(img.mask)/(img.shape[1] * img.shape[2]) > clarityThreshold
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
    return np.ma.array([pickClearImg(filterImgMskSet(imgMsk, clarityThreshold), numImgToPick=numImgToPick)
                        for imgMsk in tqdm(imgMskLR, desc=desc)])


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
    sortedIndices = np.argsort(-(np.sum(imgMsk.mask, axis=(1, 2, 3))))
    sortedImgMskArray = imgMsk[sortedIndices]
    if numImgToPick < len(imgMsk):
        trimmedImgMsk = sortedImgMskArray[:numImgToPick]
    else:
        trimmedImgMsk = np.copy(sortedImgMskArray)
        while len(trimmedImgMsk) < numImgToPick:
            shuffledIndices = np.random.choice(sortedIndices, size=len(sortedIndices), replace=False)
            toAppend = imgMsk[shuffledIndices]
            trimmedImgMsk = np.ma.concatenate((trimmedImgMsk, toAppend))
        trimmedImgMsk = trimmedImgMsk[:numImgToPick]
    return trimmedImgMsk


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
    isImageClearEnough = np.array([np.count_nonzero(img.mask)/(img.shape[1] * img.shape[2]) > clarityThreshold
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


def saveArrays(inputDictionary: Dict, parentDir: str, band: str):
    '''
    Saves numpy arrays per imageset.
    This method serves as an intermediate checkpoint for low memry users.

    inputDictionary = { 'imgLRSetsUpscaled': [],
                        'imgLRSets': [],
                        'imgHRSets': [],
                        'maskLRSetsUpscaled': [],
                        'maskLRSets': [],
                        'maskHRSets': [],
                        'names': []}
    '''
    # Define directory
    dirToSave = os.path.join(parentDir, 'numpyArrays', band)
    if not os.path.exists(dirToSave):
        os.mkdir(dirToSave)

    # Iterate through imageset arays and save them
    numSets = len(inputDictionary['names'])

    for i in tqdm(range(numSets), desc='[ INFO ] Saving numpy arrays                   '):
        np.save(os.path.join(dirToSave,
                             'imgLRSetsUpscaled_{}.npy'.format(inputDictionary['names'][i])),
                inputDictionary['imgLRSetsUpscaled'][i], allow_pickle=True)
        np.save(os.path.join(dirToSave,
                             'imgLRSets_{}.npy'.format(inputDictionary['names'][i])),
                inputDictionary['imgLRSets'][i], allow_pickle=True)
        np.save(os.path.join(dirToSave,
                             'imgHRSets_{}.npy'.format(inputDictionary['names'][i])),
                inputDictionary['imgHRSets'][i], allow_pickle=True)
        np.save(os.path.join(dirToSave,
                             'maskLRSetsUpscaled_{}.npy'.format(inputDictionary['names'][i])),
                inputDictionary['maskLRSetsUpscaled'][i], allow_pickle=True)
        np.save(os.path.join(dirToSave,
                             'maskLRSets_{}.npy'.format(inputDictionary['names'][i])),
                inputDictionary['maskLRSets'][i], allow_pickle=True)
        np.save(os.path.join(dirToSave,
                             'maskHRSets_{}.npy'.format(inputDictionary['names'][i])),
                inputDictionary['maskHRSets'][i], allow_pickle=True)


def imageSetToNumpyArrayHelper(imageSetsUpscaled: Dict, imageSets: Dict, isGrayScale: bool, isNHWC: bool):
    '''
    Helper function for imageSetToNumpyArray function.
    Iterates thru all the elemetns in the dictionary and applies the imageSetToNumpyArray function
    '''
    # Initialize Output dictionary
    output = {'imgLRSetsUpscaled': [],
              'imgLRSets': [],
              'imgHRSets': [],
              'maskLRSetsUpscaled': [],
              'maskLRSets': [],
              'maskHRSets': [],
              'names': []}

    names = []
    for name in tqdm(imageSets.keys(), desc='[ INFO ] Converting imageSets into numpy arrays'):
        currSetUpscaled = imageSetsUpscaled[name]
        ioImgPairUpscaled, ioMaskPairUpscaled = imageSetToNumpyArray(imageSet=currSetUpscaled,
                                                                     isGrayScale=isGrayScale, isNWHC=isNHWC)
        lrImgUpscaled, hrImg = ioImgPairUpscaled
        lrMaskUpscaled, hrMask = ioMaskPairUpscaled

        currSet = imageSets[name]
        ioImgPair, ioMaskPair = imageSetToNumpyArray(imageSet=currSet,
                                                     isGrayScale=isGrayScale, isNWHC=isNHWC)
        lrImg, _ = ioImgPairUpscaled
        lrMask, _ = ioMaskPairUpscaled

        output['imgLRSetsUpscaled'].append(lrImgUpscaled)
        output['maskLRSetsUpscaled'].append(lrMaskUpscaled)
        output['imgLRSets'].append(lrImg)
        output['imgHRSets'].append(hrImg)
        output['maskLRSets'].append(lrMask)
        output['maskHRSets'].append(hrMask)
        output['names'].append(name)

    return output


def generatePatchDatasetFromSavedFile(srcFolder: str, dstFolder: str, names: List[str], useUpsample: bool,
                                      patchSize: int, thresholdPatchesPerImgSet: int, thresholdClarityLR: float,
                                      thresholdClarityHR: float):
    '''
    Sample patches from the low res image.
    Patches are considered good at it is atleast n% cleared.

    Input:
    inputDictionary: Dict -> a dictionary containing the LR images and its upscaled versions, the HR images,
                             upsampleScale, names, shifts, and respective masks.
    patchSize: int -> size of patch to sample
    thresholdPatchesPerImgSet: int

    Output:
    patchesPerImgSet: Dict
    '''
    # Safety checks
    if not os.path.exists(dstFolder):
        os.mkdir(dstFolder)

    # Set maximum number of trials to get a viable Patches
    MAX_TRIAL = 100000
    PATCH_PER_SET = 9

    # Initialize outputDict
    outputDict = {}

    # Do we use the upsampled images?
    isUpsample = ''
    scale = 3
    if useUpsample:
        isUpsample = 'Upscaled'
        scale = 1

    # Extract constants
    numSets = len(names)
    sampleFname = os.path.join(srcFolder, 'imgLRSets{}_{}.npy'.format(isUpsample, names[0]))
    sampleArray = np.load(sampleFname)
    shapeUpscaled = list(sampleArray[0][0].shape)[1:]
    totalNumPixInPatch = patchSize * patchSize

    # Iterate thru all sets
    for i in tqdm(range(numSets), desc='[ INFO ] Finding patches                       '):
        # Extract relevant arrays from the inputDictionary
        currImgSetLR = loadAndRemove(os.path.join(srcFolder, 'imgLRSets{}_{}.npy'.format(isUpsample, names[i])))
        currMaskSetLR = loadAndRemove(os.path.join(srcFolder, 'maskLRSets{}_{}.npy'.format(isUpsample, names[i])))

        currImgSetHR = np.load(os.path.join(srcFolder, 'imgHRSets_{}.npy'.format(names[i])))
        currMaskSetHR = np.load(os.path.join(srcFolder, 'maskHRSets_{}.npy'.format(names[i])))

        # Initialize accumulators
        currTrial = 0
        currNumPatches = 0
        coordinatesForTheSet = []
        imgLRPatches, imgHRPatches = [], []
        maskLRPatches, maskHRPatches = [], []
        coordinates = []
        shiftsPatch = []

        # Trials to SUCCESS
        while True:
            # Define stopping condition: MAX_TRIAL is exceeded or thresholdPatchesPerImgSet is satisfied
            if currNumPatches >= thresholdPatchesPerImgSet or currTrial >= MAX_TRIAL:
                if imgLRPatches:
                    np.save(os.path.join(dstFolder, 'imgLRPatches_{}.npy'.format(names[i])),
                            np.stack(imgLRPatches), allow_pickle=True)
                    np.save(os.path.join(dstFolder, 'imgHRPatches_{}.npy'.format(names[i])),
                            np.stack(imgHRPatches), allow_pickle=True)
                    np.save(os.path.join(dstFolder, 'maskLRPatches_{}.npy'.format(names[i])),
                            np.stack(maskLRPatches), allow_pickle=True)
                    np.save(os.path.join(dstFolder, 'maskHRPatches_{}.npy'.format(names[i])),
                            np.stack(maskHRPatches), allow_pickle=True)
                    np.save(os.path.join(dstFolder, 'shifts_{}.npy'.format(names[i])),
                            np.stack(shiftsPatch), allow_pickle=True)
                break

            # Sample topleft and bottomright ccoordinates for a patch
            topLeft, btmRight = sampleCoordinates(imgSize=shapeUpscaled, patchSize=[patchSize, patchSize])
            xZero, yZero = topLeft
            xOne, yOne = btmRight

            # Extract patches using the sampled coordinates
            patchImgLR = currImgSetLR[:, :, yZero: yOne, xZero: xOne]  # [numSamples, channels, height, width]
            patchImgHR = currImgSetHR[:, :, yZero*scale: yOne*scale, xZero*scale: xOne*scale]

            patchMaskLR = currMaskSetLR[:, :, yZero: yOne, xZero: xOne] > 0  # [numSamples, channels, height, width]
            patchMaskHR = currMaskSetHR[:, :, yZero*scale: yOne*scale, xZero*scale: xOne*scale] > 0

            # Check clarity of the low resulution patches
            clearPercentageArrayLR = np.sum(patchMaskLR, axis=(1, 2, 3)) / totalNumPixInPatch
            isSampleClearLR = clearPercentageArrayLR > thresholdClarityLR
            isSampleGoodLR = np.sum(isSampleClearLR) > PATCH_PER_SET

            clearPercentageArrayHR = np.sum(patchMaskHR, axis=(1, 2, 3)) / totalNumPixInPatch
            isSampleClearHR = clearPercentageArrayHR > thresholdClarityHR
            isSampleGoodHR = np.sum(isSampleClearHR)

            if isSampleGoodLR and isSampleGoodHR:
                imgLRPatches.append(patchImgLR)
                imgHRPatches.append(patchImgHR)
                maskLRPatches.append(patchMaskLR)
                maskHRPatches.append(patchMaskHR)
                coordinatesForTheSet.append((topLeft, btmRight))
                shiftsPatch.append(shift[i])
                currNumPatches += 1

            currTrial += 1

        outputDict[names[i]] = coordinatesForTheSet

    return outputDict


def loadAndRemove(filePath):
    loadedFile = np.load(filePath, allow_pickle=True)
    os.remove(filePath)
    return loadedFile


def generatePatchDataset(inputDictionary: Dict, useUpsample: bool, patchSize: int,
                         thresholdPatchesPerImgSet: int, thresholdClarityLR: float,
                         thresholdClarityHR: float):
    '''
    Sample patches from the low res image.
    Patches are considered good at it is atleast n% cleared.

    Input:
    inputDictionary: Dict -> a dictionary containing the LR images and its upscaled versions, the HR images,
                             upsampleScale, names, shifts, and respective masks.
    patchSize: int -> size of patch to sample
    thresholdPatchesPerImgSet: int

    Output:
    patchesPerImgSet: Dict
    '''
    # Set maximum number of trials to get a viable Patches
    MAX_TRIAL = 100000
    PATCH_PER_SET = 9

    # Initialize outputDict
    outputDict = {}

    # Do we use the upsampled images?
    isUpsample = ''
    scale = inputDictionary['upsampleScale']
    if useUpsample:
        isUpsample = 'Upscaled'
        scale = 1

    # Initialize accumulators
    imgLRPatches, imgHRPatches = [], []
    maskLRPatches, maskHRPatches = [], []
    coordinates = []
    shiftsPatch = []
    names = []

    # Extract constants
    numSets = len(inputDictionary['imgLRSets' + isUpsample])
    shapeUpscaled = list(inputDictionary['imgLRSets' + isUpsample][0][0][0].shape)[1:]
    totalNumPixInPatch = patchSize * patchSize

    # Iterate thru all sets
    for i in tqdm(range(numSets), desc='[ INFO ] Finding patches                       '):
        # Extract relevant arrays from the inputDictionary
        currImgSetLR = inputDictionary['imgLRSets' + isUpsample][i]
        currMaskSetLR = inputDictionary['maskLRSets' + isUpsample][i]

        currImgSetHR = inputDictionary['imgHRSets'][i]
        currMaskSetHR = inputDictionary['maskHRSets'][i]

        currName = inputDictionary['names'][i]

        # Initialize accumulators
        currTrial = 0
        currNumPatches = 0
        coordinatesForTheSet = []

        # Trials or SUCCESS
        while True:
            # Define stopping condition: MAX_TRIAL is exceeded or thresholdPatchesPerImgSet is satisfied
            if currNumPatches >= thresholdPatchesPerImgSet or currTrial >= MAX_TRIAL:
                break

            # Sample topleft and bottomright ccoordinates for a patch
            topLeft, btmRight = sampleCoordinates(imgSize=shapeUpscaled, patchSize=[patchSize, patchSize])
            xZero, yZero = topLeft
            xOne, yOne = btmRight

            # Extract patches using the sampled coordinates
            patchImgLR = currImgSetLR[:, :, yZero: yOne, xZero: xOne]  # [numSamples, channels, height, width]
            patchImgHR = currImgSetHR[:, :, yZero*scale: yOne*scale, xZero*scale: xOne*scale]

            patchMaskLR = currMaskSetLR[:, :, yZero: yOne, xZero: xOne]  # [numSamples, channels, height, width]
            patchMaskHR = currMaskSetHR[:, :, yZero*scale: yOne*scale, xZero*scale: xOne*scale]

            # Check clarity of the low resulution patches
            clearPercentageArrayLR = np.sum(patchMaskLR, axis=(1, 2, 3)) / totalNumPixInPatch
            isSampleClearLR = clearPercentageArrayLR > thresholdClarityLR
            isSampleGoodLR = np.sum(isSampleClearLR) > PATCH_PER_SET

            clearPercentageArrayHR = np.sum(patchMaskHR, axis=(1, 2, 3)) / totalNumPixInPatch
            isSampleClearHR = clearPercentageArrayHR > thresholdClarityHR
            isSampleGoodHR = np.sum(isSampleClearHR)

            if isSampleGoodLR and isSampleGoodHR:
                imgLRPatches.append(patchImgLR)
                imgHRPatches.append(patchImgHR)
                maskLRPatches.append(patchMaskLR)
                maskHRPatches.append(patchMaskHR)
                coordinatesForTheSet.append((topLeft, btmRight))
                shiftsPatch.append(shift[i])
                names.append(currName)
                currNumPatches += 1

            currTrial += 1

        coordinates.append(coordinatesForTheSet)

    # Append to outputDict
    outputDict['imgPatchesLR'] = imgLRPatches
    outputDict['maskPatchesLR'] = maskLRPatches
    outputDict['imgPatchesHR'] = imgHRPatches
    outputDict['maskPatchesHR'] = maskHRPatches
    outputDict['shifts'] = shiftsPatch
    outputDict['coordinates'] = coordinates
    outputDict['names'] = names

    return outputDict


def sampleCoordinates(imgSize: List[int], patchSize: List[int]):
    '''
    Sample a random patch with size patchSize in imgSize!

    Input:
    imgSize: List[int] -> size of the image to patch sample from.
    patchSize: List[int] -> size of patch to sample.

    Output:
    topLeftXYCoordinates, btmRightXYCoordinates
    '''
    topLeftX = random.randint(0, imgSize[0] - patchSize[0] - 1)
    topLeftY = random.randint(0, imgSize[1] - patchSize[1] - 1)
    btmRightX = topLeftX + patchSize[0]
    btmRightY = topLeftY + patchSize[1]

    return (topLeftX, topLeftY), (btmRightX, btmRightY)


def generateDataDir(isTrainData: bool, NIR: bool):
    '''
    Generate a list containing the directories of test/train data

    Input:
    isTrainData: bool -> True if train data
            NIR: bool -> True if NIR band
    Output:
    List that contains string of the form 'imgsetxxxxx'
    '''
    band = 'NIR' if NIR else 'RED'
    dataType = 'train' if isTrainData else 'test'
    imageDir = os.path.join(DATA_BANK_DIRECTORY, dataType, band)
    dirList = sorted([os.path.basename(x) for x in glob.glob(imageDir + '/imgset*')])
    # dirList = dirList[:25]
    return dirList


def generateNormArray(dirList: List[str]):
    '''
    Generate norm array to be used for score calculation

    Output:
    np.array of norms from norm.csv
    '''
    csvDir = os.path.join(DATA_BANK_DIRECTORY, 'norm.csv')
    dataFrame = pd.read_csv(csvDir, sep=' ', header=None)
    dataFrame.columns(['dataset', 'norm'])
    norm = dataFrame.loc[dataFrame['dataset'].isin(dirList)]['norm'].values
    return norm


def generateImageSetDict(imageSetNum: str, isTrainData: bool, NIR: bool):
    '''
    Generate a tuple of dictionary (ImageArrayDict, ImageMaskDict)

    Input:
    imageSetNum: str  -> folder name of the image scene
    isTrainData: bool -> True if train data
            NIR: bool -> True if NIR band

    Output:
    A tuple (ImageArrayDict, ImageMaskDict)
    '''
    # Initialize outputs
    imgMask = {}
    imgArray = {}

    # Initialize Data Path
    band = 'NIR' if NIR else 'RED'
    dataType = 'train' if isTrainData else 'test'
    imageDir = os.path.join(DATA_BANK_DIRECTORY, dataType, band, imageSetNum)
    if not os.path.exists(imageDir):
        print("Path does not exist")
        return ({}, {})

    # Iterate through all items and populate the Dicts
    imageList = sorted(os.listdir(imageDir))
    if isTrainData:
        imgArray['HR'] = io.imread(os.path.join(imageDir, 'HR.png'))
        imgMask['HR'] = io.imread(os.path.join(imageDir, 'SM.png'))
    for img in imageList:
        if img == 'HR.png':
            continue
        elif img == 'SM.png':
            continue
        elif img[0:2] == 'LR':
            imgArray[img.split('.')[0]] = io.imread(os.path.join(imageDir, img))
        else:
            imgMask['LR' + img.split('.')[0][2:]] = io.imread(os.path.join(imageDir, img)).astype(np.bool)

    return (imgArray, imgMask)


def generateImageSet(isTrainData: bool, NIR: bool):
    '''
    Generate a dictionary with the key with the form 'imgsetxxxxx' and values
    as tuple (ImageArrayDict, ImageMaskDict)
    '''

    dirList = generateDataDir(isTrainData, NIR)
    imageSet = {imgSet: generateImageSetDict(imgSet, isTrainData, NIR)
                for imgSet in tqdm(dirList, desc='[ INFO ] Loading data into a dictionary        ')}
    return imageSet


def removeImageWithOutlierPixels(imageSet: Dict, threshold: int, isTrainData: bool):
    '''
    This function removes images with pixels greater than the assigned threshold.
    Images are 14 bits in depth but is represented by 16 bit array.
    We use threshold around 32000 ~ 60000.

    Input:
    imageSet: Dict[imgsetxxxx] = Tuple(ImageArrayDict, ImageMaskDict)
    threshold: int

    Output: Dict
    '''
    # Initialize info list
    imgSetLRPair = []
    imageSetRemove = []

    # If isTrainData set threshold to 9 + 1 (+1 for the HR image)
    numImagesThreshold = 10 if isTrainData else 9

    # Iterate through all arrays and determine if they have outliers
    for keySet in tqdm(list(imageSet.keys()), desc='[ INFO ] Removing outliers in dataset          '):
        imgArrayDict, imgMaskDict = imageSet[keySet]
        for keyArray in list(imgArrayDict.keys()):
            # Remove images with high pixel values
            if keyArray == 'HR':
                continue
            if (imgArrayDict[keyArray] > threshold).any():
                imgSetLRPair.append((keySet, keyArray))
                del imgArrayDict[keyArray]
                del imgMaskDict[keyArray]
        # Remove images with LR images below 9
        if len(imgArrayDict.keys()) < numImagesThreshold:
            imageSetRemove.append(keySet)
            del imageSet[keySet]

    if DEBUG:
        print('imgSet and LR image pair to be removed are as follows \n{} \n \
               imageSet to be removed are as follows \n{}'.format(imgSetLRPair, imageSetRemove))

    return imageSet


def upsampleImages(imageSets: Dict, scale: int):
    '''
    Converts all images and its masks from 128x128 -> 384x384 by upsampling.

    Input:
    imageSet: Dict -> imageSet[imagesetxxxx] = Tuple(ImageArrayDict, ImageMaskDict)
       scale: int

    Output:
    imageSet: Dict
    '''
    # Iterate for all imageSet
    for keySet in tqdm(imageSets.keys(), desc='[ INFO ] Upscaling LowRes images               '):
        imgArrayDict, imgMaskDict = imageSets[keySet]

        if DEBUG:
            print('[ INFO ] Processing {}...'.format(keySet))

        # Iterate for all LR images
        for keyArray in imgArrayDict.keys():
            # Skip the HR images
            if keyArray == 'HR':
                continue
            # Rescale LR images
            imgArrayDict[keyArray] = rescale(imgArrayDict[keyArray],
                                             scale=scale,
                                             order=3,  # bicubic interpolation
                                             mode='edge',
                                             anti_aliasing=False,
                                             multichannel=False,
                                             preserve_range=True)
            imgArrayDict[keyArray] = imgArrayDict[keyArray].astype('float32')

            # Rescale corresponding masks
            imgMaskDict[keyArray] = rescale(imgMaskDict[keyArray],
                                            scale=scale,
                                            order=0,
                                            mode='constant',
                                            anti_aliasing=False,
                                            multichannel=False,
                                            preserve_range=True)
            imgMaskDict[keyArray] = imgMaskDict[keyArray].astype('bool')

            if DEBUG:
                print('[ INFO ] Image size upscaled to {}x{}.'
                      .format(imgArrayDict[keyArray].shape[0], imgArrayDict[keyArray].shape[1]))

        # Reassign imageSet
        imageSets[keySet] = tuple([imgArrayDict, imgMaskDict])
        if DEBUG:
            print('[ SUCCESS ] {} upscaled.'.format(keySet))

    return imageSets


def imageSetToNumpyArray(imageSet: Tuple, isGrayScale: bool, isNWHC: bool):
    '''
    This function takes in the imageSet dictionary and
    transforms it to a Tuple of 4D numpy array with dimensions
    ([numLRImgs, imgHeight, imgWidth, channels], [1, imgHeight, imgWidth, channels]).
    This is done with the image array and the mask array.

    Note that the first element of the numpy array is the input and
    the second one is the expected output of the network.

    Input:
    imageSet: Tuple   -> A tuple of dictionary that take the form (ImageArrayDict, ImageMaskDict)
    isGrayScale: bool -> Indicator if grayscale
    isNHWC: bool      -> Indicator if the desired output is of the form
                         [numLRImgs, imgHeight, imgWidth, channels] or
                         [numLRImgs, channels, imgHeight, imgWidth]

    Output:
    A pair of tuples [Input, Output]
    For each tupple has two elements, one for image and one for the mask.
    '''
    # Extract valuable constants
    imgArrayDict, imgMaskDict = imageSet
    numLowResImg = len(imgArrayDict) - 1
    numHighResImg = 1
    if isGrayScale:
        heightHighRes, widthHighRes = imgArrayDict['HR'].shape
        heightLowRes, widthLowRes = imgArrayDict[list(imgArrayDict.keys())[1]].shape
        channel = 1
    else:
        heightHighRes, widthHighRes, channel = imgArrayDict['HR'].shape
        heightLowRes, widthLowRes, channel = imgArrayDict[list(imgArrayDict.keys())[1]].shape

    # Initialize numpy arrays
    imgLowResArray = np.zeros((numLowResImg, channel, heightLowRes, widthLowRes))
    maskLowResArray = np.zeros((numLowResImg, channel, heightLowRes, widthLowRes))
    imgHighResArray = np.zeros((numHighResImg, channel, heightHighRes, widthHighRes))
    maskHighResArray = np.zeros((numHighResImg, channel, heightHighRes, widthHighRes))

    # Remove HR image and mask from the dictionary
    if isGrayScale:
        imgHighResArray[0, :, :, :] = np.array([imgArrayDict['HR']])
        maskHighResArray[0, :, :, :] = np.array([imgMaskDict['HR']])
    else:
        imgHighResArray[0, :, :, :] = imgArrayDict['HR']
        maskHighResArray[0, :, :, :] = imgMaskDict

    i = 0
    for keyArray in imgArrayDict.keys():
        if keyArray == 'HR':
            continue
        imgArray, imgMask = imgArrayDict[keyArray], imgMaskDict[keyArray]
        if isGrayScale:
            imgLowResArray[i, :, :, :] = np.array([imgArray])
            maskLowResArray[i, :, :, :] = np.array([imgMask])
        else:
            imgLowResArray[i, :, :, :] = imgArray
            maskLowResArray[i, :, :, :] = imgMask
        # increment
        i += 1

    # Reshape to [numLRImgs, imgHeight, imgWidth, channels]
    if isNWHC:
        imgLowResArray = imgLowResArray.transpose((0, 2, 3, 1))
        maskLowResArray = maskLowResArray.transpose((0, 2, 3, 1))
        imgHighResArray = imgHighResArray.transpose((0, 2, 3, 1))
        maskHighResArray = maskHighResArray.transpose((0, 2, 3, 1))

    # Delete for memory saving
    del imageSet

    return (imgLowResArray, imgHighResArray), (maskLowResArray, maskHighResArray)


def correctShifts(inputDictionary: Dict, upsampleScale: int):
    '''
    As per the data website the low resolution images are not adjusted for its shift.
    We adjust the the low resolution images for the shift.
    We calculate the shift of every low resolution image with respect to the most clear
    low resolution image. That is the sum of all elements of its mask is the highest.

    Input:
    imageSets: np.ndarray -> A list of 4D array with dimensions [numLRImgs, channels, imgHeight, imgWidth]
    maskSets: np.ndarray  -> A list of 4D array with dimensions [numLRMasks, channels, imgHeight, imgWidth]
    upsampleScale: int    -> The scale factor for which was used in the upsampling method.
    output = {'imgLRSetsUpscaled': [],
              'imgLRSets': [],
              'imgHRSets': [],
              'maskLRSetsUpscaled': [],
              'maskLRSets': [],
              'maskHRSets': [],
              'names': []}

    Output:
    Corrected and trimmed version  of the input dataset
    '''
    # Extract constants
    numSets = len(inputDictionary['imgLRSetsUpscaled'])

    # Initialize outputs
    output = {}
    output['newSortedImageSets'] = []
    output['newSortedMaskSets'] = []
    output['trimmedImageSetsTrans'] = []
    output['trimmedMaskSetsTrans'] = []
    output['trimmedImageSetsOrig'] = []
    output['trimmedMaskSetsOrig'] = []
    output['shifts'] = []

    # Iterate thru all image sets
    for i in tqdm(range(numSets), desc='[ INFO ] Correcting shifts in imagesets        '):
        imgSetTrans, maskSetTrans = inputDictionary['imgLRSetsUpscaled'][i], inputDictionary['maskLRSetsUpscaled'][i]
        imgSetOrig, maskSetOrig = inputDictionary['imgLRSets'][i], inputDictionary['maskLRSets'][i]

        numLRImgs = imgSetTrans.shape[0]
        sortedIdx = np.argsort(np.sum(maskSetTrans, axis=(1, 2, 3)))[::-1]  # descending order

        imgSetTrans = imgSetTrans[sortedIdx, :, :, :]
        maskSetTrans = maskSetTrans[sortedIdx, :, :, :]
        imgSetOrig = imgSetOrig[sortedIdx, :, :, :]
        maskSetOrig = maskSetOrig[sortedIdx, :, :, :]

        output['newSortedImageSets'].append(imgSetTrans)
        output['newSortedMaskSets'].append(maskSetTrans)

        referenceImage = imgSetTrans[0, :, :, :]  # most clear image
        referenceMask = maskSetTrans[0, :, :, :]  # highest cummulative sum
        origImage = imgSetOrig[0, :, :, :]
        origMask = maskSetOrig[0, :, :, :]

        # Copy arrays for stacking
        trimmedImageSetTrans = [np.array([np.copy(referenceImage)])]
        trimmedMaskSetTrans = [np.array([np.copy(referenceMask)])]
        trimmedImageSetOrig = [np.array([np.copy(origImage)])]
        trimmedMaskSetOrig = [np.array([np.copy(origMask)])]

        # Number of LR images included
        counter = 1

        # Initialize setShift accumulator
        setShift = []

        # Iterate thru all LR image for the current scene
        # and adjust the shift wrt the reference image
        for j in range(1, numLRImgs):
            # Initialize current images and mask
            currImage = imgSetTrans[j, :, :, :]
            currMask = maskSetTrans[j, :, :, :]

            # Calculate shift
            shift, error, diffPhase = register_translation(
                referenceImage.squeeze(), currImage.squeeze(), upsampleScale)
            shift = np.asarray(shift)

            # Skip those images with 4 shifts and above
            if (np.abs(shift) > 4).any():
                continue

            # Accumulate good shifts
            setShift.append(shift)

            # Get shapes -> format may be CWH or WHC
            x, y, z = currImage.shape

            # Correct image in the frequency domain
            correctedImageInFreqDomain = fourier_shift(np.fft.fftn(currImage.squeeze()), shift)
            correctedImage = np.fft.ifftn(correctedImageInFreqDomain)
            correctedImage = correctedImage.reshape((x, y, z))

            # Correct image in the frequency domain
            correctedMaskInFreqDomain = fourier_shift(np.fft.fftn(currMask.squeeze()), shift)
            correctedMask = np.fft.ifftn(correctedMaskInFreqDomain)
            correctedMask = correctedMask.reshape((x, y, z))

            # Stack to the reference iamge
            trimmedImageSetTrans.append(np.array([correctedImage]))
            trimmedMaskSetTrans.append(np.array([correctedMask]))
            trimmedImageSetOrig.append(np.array([imgSetOrig[j, :, :, :]]))
            trimmedMaskSetOrig.append(np.array([maskSetOrig[j, :, :, :]]))
            counter += 1

        # Remove imagesets with LR images less than 9
        if counter < 9:
            if DEBUG:
                print('An image set has been remove due to low LR image number.')
            # Remove HR Image and its mask
            del inputDictionary['imgHRSets'][i]
            del inputDictionary['maskHRSets'][i]
            del inputDictionary['names'][i]
            continue

        # shift to another big array of shifts goddammit
        output['shifts'].append(setShift)

        # Append to trimmed list
        output['trimmedImageSetsTrans'].append(np.stack(trimmedImageSetTrans))
        output['trimmedMaskSetsTrans'].append(np.stack(trimmedMaskSetTrans))
        output['trimmedImageSetsOrig'].append(np.stack(trimmedImageSetOrig))
        output['trimmedMaskSetsOrig'].append(np.stack(trimmedMaskSetOrig))
        output['names'] = inputDictionary['names']
        output['imgHRSets'] = inputDictionary['imgHRSets']
        output['maskHRSets'] = inputDictionary['maskHRSets']

    return output


def correctShiftsFromSavedArrays(folderPath: str, outputDir: str, names: List[str], upsampleScale: int):
    '''
    As per the data website the low resolution images are not adjusted for its shift.
    We adjust the the low resolution images for the shift.
    We calculate the shift of every low resolution image with respect to the most clear
    low resolution image. That is the sum of all elements of its mask is the highest.

    Input:
    imageSets: np.ndarray -> A list of 4D array with dimensions [numLRImgs, channels, imgHeight, imgWidth]
    maskSets: np.ndarray  -> A list of 4D array with dimensions [numLRMasks, channels, imgHeight, imgWidth]
    upsampleScale: int    -> The scale factor for which was used in the upsampling method.
    output = {'imgLRSetsUpscaled': [],
              'imgLRSets': [],
              'imgHRSets': [],
              'maskLRSetsUpscaled': [],
              'maskLRSets': [],
              'maskHRSets': [],
              'names': []}

    Output:
    Corrected and trimmed version  of the input dataset
    '''
    # Safety checks
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    # Extract constants
    numSets = len(names)

    # Names to Delete
    delNames = []

    # Iterate thru all image sets
    for i in tqdm(range(numSets), desc='[ INFO ] Correcting shifts in imagesets        '):
        # Load arrays from file
        imgSetTrans = loadAndRemove(os.path.join(folderPath, 'imgLRSetsUpscaled_{}.npy'.format(names[i])))
        maskSetTrans = loadAndRemove(os.path.join(folderPath, 'maskLRSetsUpscaled_{}.npy'.format(names[i])))
        imgSetOrig = loadAndRemove(os.path.join(folderPath, 'imgLRSets_{}.npy'.format(names[i])))
        maskSetOrig = loadAndRemove(os.path.join(folderPath, 'maskLRSets_{}.npy'.format(names[i])))

        numLRImgs = imgSetTrans.shape[0]
        sortedIdx = np.argsort(np.sum(maskSetTrans, axis=(1, 2, 3)))[::-1]  # descending order

        imgSetTrans = imgSetTrans[sortedIdx, :, :, :]
        maskSetTrans = maskSetTrans[sortedIdx, :, :, :]
        imgSetOrig = imgSetOrig[sortedIdx, :, :, :]
        maskSetOrig = maskSetOrig[sortedIdx, :, :, :]

        referenceImage = imgSetTrans[0, :, :, :]  # most clear image
        referenceMask = maskSetTrans[0, :, :, :]  # highest cummulative sum
        origImage = imgSetOrig[0, :, :, :]
        origMask = maskSetOrig[0, :, :, :]

        # Copy arrays for stacking
        trimmedImageSetTrans = [np.array([np.copy(referenceImage)])]
        trimmedMaskSetTrans = [np.array([np.copy(referenceMask)])]
        trimmedImageSetOrig = [np.array([np.copy(origImage)])]
        trimmedMaskSetOrig = [np.array([np.copy(origMask)])]

        # Number of LR images included
        counter = 1

        # Initialize setShift accumulator
        setShift = []

        # Iterate thru all LR image for the current scene
        # and adjust the shift wrt the reference image
        for j in range(1, numLRImgs):
            # Initialize current images and mask
            currImageUp = imgSetTrans[j, :, :, :]
            currMaskUp = maskSetTrans[j, :, :, :]
            currImage = imgSetOrig[j, :, :, :]
            currMask = maskSetOrig[j, :, :, :]

            # Calculate shift
            # shift, error, diffPhase = register_translation(
            #     referenceImage.squeeze(), currImage.squeeze(), upsampleScale)
            shiftValueUp = masked_register_translation(currImageUp, currMaskUp > 0, referenceImage)
            shiftValue = masked_register_translation(currImage, currMask > 0, origImage)

            # Skip those images with 4 shifts and above
            # if (np.abs(shiftValue) > 4).any():
            #     continue

            # Accumulate good shifts
            setShift.append(shiftValue)

            # Get shapes -> format may be CWH or WHC
            xOne, yOne, zOne = currImage.shape
            xTwo, yTwo, zTwo = currImageUp.shape

            # Correct images and masks
            correctedImage = shift(currImage, shiftValue, mode='reflect')
            correctedImage = correctedImage.reshape((xOne, yOne, zOne))

            correctedImageUp = shift(currImageUp, shiftValueUp, mode='reflect')
            correctedImageUp = correctedImageUp.reshape((xTwo, yTwo, zTwo))

            correctedMask = shift(currMask, shiftValue, mode='constant', cval=0)
            correctedMask = correctedMask.reshape((xOne, yOne, zOne))

            correctedMaskUp = shift(currMaskUp, shiftValueUp, mode='constant', cval=0)
            correctedMaskUp = correctedMaskUp.reshape((xTwo, yTwo, zTwo))

            # Stack to the reference iamge
            trimmedImageSetTrans.append(np.array([correctedImageUp]))
            trimmedMaskSetTrans.append(np.array([correctedMaskUp]))
            trimmedImageSetOrig.append(np.array([correctedImage]))
            trimmedMaskSetOrig.append(np.array([correctedMask]))
            counter += 1

        # Remove imagesets with LR images less than 9
        if counter < 9:
            if DEBUG:
                print('An image set has been remove due to low LR image number.')
            # Remove HR Image and its mask
            delNames.append(names[i])
            continue

        # shift to another big array of shifts goddammit
        np.save(os.path.join(outputDir,
                             'shifts_{}.npy'.format(names[i])), np.array(setShift), allow_pickle=True)

        # Save each sets
        np.save(os.path.join(outputDir,
                             'imgLRSetsUpscaled_{}.npy'.format(names[i])), np.stack(trimmedImageSetTrans),
                allow_pickle=True)
        np.save(os.path.join(outputDir,
                             'maskLRSetsUpscaled_{}.npy'.format(names[i])), np.stack(trimmedMaskSetTrans),
                allow_pickle=True)
        np.save(os.path.join(outputDir,
                             'imgLRSets_{}.npy'.format(names[i])), np.stack(trimmedImageSetOrig),
                allow_pickle=True)
        np.save(os.path.join(outputDir,
                             'maskLRSets_{}.npy'.format(names[i])), np.stack(trimmedMaskSetOrig),
                allow_pickle=True)
        # Copy HR arrays
        move(os.path.join(folderPath, 'imgHRSets_{}.npy'.format(names[i])),
             os.path.join(outputDir,
                          'imgHRSets_{}.npy'.format(names[i])))
        move(os.path.join(folderPath, 'maskHRSets_{}.npy'.format(names[i])),
             os.path.join(outputDir,
                          'maskHRSets_{}.npy'.format(names[i])))

    [names.remove(delName) for delName in delNames]


if __name__ == '__main__':
    opt = parser()
    main()
