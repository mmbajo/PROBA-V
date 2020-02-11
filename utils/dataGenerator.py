from typing import List

import os
import glob
import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import rescale

# Set the data data directory
# Download data at https://kelvins.esa.int/proba-v-super-resolution/data/
DATA_BANK_DIRECTORY = '/home/mark/DataBank/probav_data/'


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
                for imgSet in dirList}
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
    for keySet in imageSet.keys():
        imgArrayDict, imgMaskDict = imageSet[keySet]
        for keyArray in imgArrayDict.keys():
            # Remove images with high pixel values
            if keyArray == 'HR':
                continue
            if (imgArrayDict[keyArray] < threshold).any():
                imgSetLRPair.append((keySet, keyArray))
                del imgArrayDict[keyArray]
                del imgMaskDict[keyArray]
        # Remove images with LR images below 9
        if len(imageSet[keySet]) < numImagesThreshold:
            imageSetRemove.append(keySet)
            del imageSet[keySet]

    print('imgSet and LR image pair to be removed are as follows \n{} \n \
           imageSet to be removed are as follows \n{}'.format(imgSetLRPair, imageSetRemove))

    return imageSet


def upsampleImages(imageSet: Dict, scale: int):
    '''
    Converts all images and its masks from 128x128 -> 384x384 by upsampling.

    Input:
    imageSet: Dict -> imageSet[imagesetxxxx] = Tuple(ImageArrayDict, ImageMaskDict)
       scale: int

    Output:
    imageSet: Dict
    '''
    # Iterate for all imageSet
    for keySet in imageSet.keys():
        imgArrayDict, imgMaskDict = imageSet[keySet]

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

        # Reassign imageSet
        imageSet[keySet] = tuple(imgArrayDict, imgMaskDict)

    return imageSet


def correctShifts(imageSet: Dict):
    pass
