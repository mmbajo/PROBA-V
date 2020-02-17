from typing import List, Tuple, Dict
import argparse

import os
import glob
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from shutil import move
from skimage import io
from skimage.transform import rescale
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift

DEBUG = 0
# Set the data data directory
# Download data at https://kelvins.esa.int/proba-v-super-resolution/data/
DATA_BANK_DIRECTORY = '/home/mark/DataBank/probav_data/'
DATA_BANK_DIRECTORY_PREPROCESSING_CHKPT = '/home/mark/DataBank/PROBA-V-CHKPT'


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--band', default='NIR', type=str)
    parser.add_argument('--dir', type=str, default=DATA_BANK_DIRECTORY)
    parser.add_argument('--split', type=float, default=0.7)
    opt = parser.parse_args()
    return opt


def main(opt):
    # Parse inputs
    NIR = True if opt.band == 'NIR' else False

    # Load dataset
    imgAllSets = generateImageSet(isTrainData=True, NIR=NIR)

    # Remove outliers
    imgAllSets = removeImageWithOutlierPixels(imageSet=imgAllSets, threshold=60000, isTrainData=True)

    # Upscale
    imgAllSetsUpScaled = upsampleImages(imageSets=imgAllSets, scale=3)  # 128x128 -> 384x384

    # Convert dataset to numpy array
    output = imageSetToNumpyArrayHelper(imgAllSetsUpScaled, imgAllSets, isGrayScale=True, isNHWC=False)

    # Intermediate writing of numpy arrays
    # output = {'imgLRSetsUpscaled': [],
    #           'imgLRSets': [],
    #           'imgHRSets': [],
    #           'maskLRSetsUpscaled': [],
    #           'maskLRSets': [],
    #           'maskHRSets': [],
    #           'names': []}
    saveArrays(output, DATA_BANK_DIRECTORY_PREPROCESSING_CHKPT)

    # Get shifts and trim dataset based on shift threshold
    # Read more about this shifts happening in LR Images here
    # https://kelvins.esa.int/proba-v-super-resolution/data/
    # output = correctShifts(output, upsampleScale=1)
    correctShiftsFromSavedArrays(os.path.join(DATA_BANK_DIRECTORY_PREPROCESSING_CHKPT,
                                              'numpyArrays', opt.band),
                                 os.path.join(DATA_BANK_DIRECTORY_PREPROCESSING_CHKPT,
                                              'correctedShifts', opt.band),
                                 output['names'], upsampleScale=1)

    # Add key value element in output dictionary
    output['upsampleScale'] = 3

    # Create Patches
    output = generatePatchDataset(output, useUpsample=True, patchSize=96,
                                  thresholdPatchesPerImgSet=9, thresholdClarityLR=0.7,
                                  thresholdClarityHR=0.85)

    # Return a list of input outputs (maybe a 5D numpy array)
    normArray = generateNormArray(dirList=output['names'])

    # Save to file for training


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
                inputDictionary['imgLRSetsUpscaled'][i])
        np.save(os.path.join(dirToSave,
                             'imgLRSets_{}.npy'.format(inputDictionary['names'][i])),
                inputDictionary['imgLRSets'][i])
        np.save(os.path.join(dirToSave,
                             'imgHRSets_{}.npy'.format(inputDictionary['names'][i])),
                inputDictionary['imgHRSets'][i])
        np.save(os.path.join(dirToSave,
                             'maskLRSetsUpscaled_{}.npy'.format(inputDictionary['names'][i])),
                inputDictionary['maskLRSetsUpscaled'][i])
        np.save(os.path.join(dirToSave,
                             'maskLRSets_{}.npy'.format(inputDictionary['names'][i])),
                inputDictionary['maskLRSets'][i])
        np.save(os.path.join(dirToSave,
                             'maskHRSets_{}.npy'.format(inputDictionary['names'][i])),
                inputDictionary['maskHRSets'][i])


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
        key = 'Upscaled'
        scale = 1

    # Extract constants
    numSets = len(names)
    sampleFname = os.path.join(srcFolder, 'imgLRSets{}_{}.npy'.format(isUpsample, names[0]))
    sampleArray = np.load(sampleFname)
    shapeUpscaled = list(sampleArray[0][0].shape)[1:]
    totalNumPixInPatch = patchSize * patchSize

    # Iterate thru all sets
    for i in tqdm(range(numSets), desc='[ INFO ] Upscaling LowRes images               '):
        # Extract relevant arrays from the inputDictionary
        currImgSetLR = loadAndRemove(os.path.join(srcFolder, 'imgLRSets{}_{}.npy'.format(isUpsample, names[i])))
        currMaskSetLR = loadAndRemove(os.path.join(srcFolder, 'maskLRSets{}_{}.npy'.format(isUpsample, names[i])))

        currImgSetHR = loadAndRemove(os.path.join(srcFolder, 'imgHRSets_{}.npy'.format(names[i])))
        currMaskSetHR = loadAndRemove(os.path.join(srcFolder, 'maskHRSets_{}.npy'.format(names[i])))

        # Initialize accumulators
        currTrial = 0
        currNumPatches = 0
        coordinatesForTheSet = []
        imgLRPatches, imgHRPatches = [], []
        maskLRPatches, maskHRPatches = [], []
        coordinates = []
        shiftsPatch = []

        # Trials or SUCCESS
        while True:
            # Define stopping condition: MAX_TRIAL is exceeded or thresholdPatchesPerImgSet is satisfied
            if currNumPatches >= thresholdPatchesPerImgSet or currTrial >= MAX_TRIAL:
                if imgLRPatches:
                    np.save(os.path.join(dstFolder, 'imgLRPatches_{}.npy'.format(names[i])),
                            np.stack(imgLRPatches))
                    np.save(os.path.join(dstFolder, 'imgHRPatches_{}.npy'.format(names[i])),
                            np.stack(imgHRPatches))
                    np.save(os.path.join(dstFolder, 'maskLRPatches_{}.npy'.format(names[i])),
                            np.stack(maskLRPatches))
                    np.save(os.path.join(dstFolder, 'maskHRPatches_{}.npy'.format(names[i])),
                            np.stack(maskHRPatches))
                    np.save(os.path.join(dstFolder, 'shifts_{}.npy'.format(names[i])),
                            np.stack(shiftsPatch))
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
                currNumPatches += 1

            currTrial += 1

        outputDict[names[i]] = coordinatesForTheSet

    return outputDict


def loadAndRemove(filePath):
    loadedFile = np.load(filePath)
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
        key = 'Upscaled'
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
    for i in tqdm(range(numSets), desc='[ INFO ] Upscaling LowRes images               '):
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
            delNames.append(names[i])
            continue

        # shift to another big array of shifts goddammit
        np.save(os.path.join(outputDir,
                             'shifts_{}.npy'.format(names[i])), np.array(setShift))

        # Save each sets
        np.save(os.path.join(outputDir,
                             'imgLRSetsUpscaled_{}.npy'.format(names[i])), np.stack(trimmedImageSetTrans))
        np.save(os.path.join(outputDir,
                             'maskLRSetsUpscaled_{}.npy'.format(names[i])), np.stack(trimmedMaskSetTrans))
        np.save(os.path.join(outputDir,
                             'imgLRSets_{}.npy'.format(names[i])), np.stack(trimmedImageSetOrig))
        np.save(os.path.join(outputDir,
                             'maskLRSets_{}.npy'.format(names[i])), np.stack(trimmedMaskSetOrig))
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
    main(opt)
