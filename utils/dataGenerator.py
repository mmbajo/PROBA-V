from typing import List, Tuple

import os
import glob
import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import rescale
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift

DEBUG = 1
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
    for keySet in imageSets.keys():
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
        imageSets[keySet] = tuple(imgArrayDict, imgMaskDict)
        if DEBUG:
            print('[ SUCCESS ] {} upscaled.'.format(keySet))

    return imageSets


def imageSet2NumpyArray(imageSet: Tuple, isGrayScale: bool, isNWHC: bool):
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
        heightLowRes, widthLowRes = imgArrayDict[imgArrayDict.keys()[1]].shape
        channel = 1
    else:
        heightHighRes, widthHighRes, channel = imgArrayDict['HR'].shape
        heightLowRes, widthLowRes, channel = imgArrayDict[imgArrayDict.keys()[1]].shape

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

    # Delete HR arrays from the dictionary
    del imgArrayDict['HR']
    del imgMaskDict['HR']

    for i, keyArray in enumerate(imgArrayDict.keys()):
        imgArray, imgMask = imgArrayDict[keyArray], imgMaskDict[keyArray]
        if isGrayScale:
            imgLowResArray[i, :, :, :] = np.array([imgArray])
            maskLowResArray[i, :, :, :] = np.array([imgMask])
        else:
            imgLowResArray[i, :, :, :] = imgArray
            maskLowResArray[i, :, :, :] = imgMask

    # Reshape to [numLRImgs, imgHeight, imgWidth, channels]
    if isNWHC:
        imgLowResArray = imgLowResArray.transpose((0, 2, 3, 1))
        maskLowResArray = maskLowResArray.transpose((0, 2, 3, 1))
        imgHighResArray = imgHighResArray.transpose((0, 2, 3, 1))
        maskHighResArray = maskHighResArray.transpose((0, 2, 3, 1))

    # Delete for memory saving
    del imageSet

    return (imgLowResArray, imgHighResArray), (maskLowResArray, maskHighResArray)


def correctShifts(imageSetsLR: List[np.ndarray], maskSetsLR: List[np.ndarray],
                  imageSetsHR: List[np.ndarray], maskSetsHR: List[np.ndarray], upsampleScale: int):
    '''
    As per the data website the low resolution images are not adjusted for its shift.
    We adjust the the low resolution images for the shift.
    We calculate the shift of every low resolution image with respect to the most clear
    low resolution image. That is the sum of all elements of its mask is the highest.

    Input:
    imageSets: np.ndarray -> A list of 4D array with dimensions [numLRImgs, channels, imgHeight, imgWidth]
    maskSets: np.ndarray  -> A list of 4D array with dimensions [numLRMasks, channels, imgHeight, imgWidth]
    upsampleScale: int    -> The scale factor for which was used in the upsampling method.

    Output:
    Corrected and trimmed version  of the input dataset
    '''
    # Extract constants
    numSets = len(imageSetsLR)

    # Initialize outputs
    newSortedImageSets = []
    newSortedMaskSets = []
    trimmedImageSets = []
    trimmedMaskSets = []

    # Iterate thru all image sets
    for i in range(numSets):
        imgSet, maskSet = imageSetsLR[i], maskSetsLR[i]
        numLRImgs = imgSet.shape[0]
        sortedIdx = np.argsort(np.sum(maskSet, axis=(1, 2, 3)))[::-1]  # descending order

        imgSet = imgSet[sortedIdx, :, :, :]
        maskSet = maskSet[sortedIdx, :, :, :]

        newSortedImageSets.append(imgSet)
        newSortedMaskSets.append(maskSet)

        referenceImage = imgSet[0, :, :, :]  # most clear image
        referenceMask = maskSet[0, :, :, :]  # highest cummulative sum

        # Copy arrays for stacking
        trimmedImageSet = np.array([np.copy(referenceImage)])
        trimmedMaskSet = np.array([np.copy(referenceMask)])

        # Number of LR images included
        counter = 1

        # Iterate thru all LR image for the current scene
        # and adjust the shift wrt the reference image
        for j in range(1, numLRImgs):
            # Initialize current images and mask
            currImage = imgSet[j, :, :, :]
            currMask = maskSet[j, :, :, :]

            # Calculate shift
            shift, error, diffPhase = register_translation(
                referenceImage.squeeze(), currImage.squeeze(), upsampleScale)
            shift = np.asarray(shift)

            # Skip those images with 4 shifts and above
            if (np.abs(shift) > 4).any():
                continue

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
            trimmedImageSet = np.stack(trimmedImageSet, np.array([correctedImage]))
            trimmedMaskSet = np.stack(trimmedMaskSet, np.array([correctedMask]))
            counter += 1

        # Remove imagesets with LR images less than 9
        if counter < 9:
            print('An image set has been remove due to low LR image number.')
            # Remove HR Image and its mask
            del imageSetsHR[i]
            del maskSetsHR[i]
            continue

        # Append to trimmed list
        trimmedImageSets.append(trimmedImageSet)
        trimmedMaskSets.append(trimmedMaskSet)

        return trimmedImageSets, imageSetsHR, trimmedMaskSets, maskSetsHR


def main():
    # Load dataset
    # Remove outliers
    # Upscale
    # Remove outliers
    # Convert to numpy array
    # Correct shifts
    # Return a list of input outputs (maybe a 5D numpy array)
    pass


if __name__ == '__main__':
    main()
