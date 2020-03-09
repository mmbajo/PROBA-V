from typing import List

import argparse
import os
import numpy as np
from skimage import io
import tensorflow as tf
from tqdm import trange
import matplotlib.pyplot as plt
import tensorflow as tf
from models.loss import Losses
from utils.dataGenerator import generatePatchesPerImgSet


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, default='evaluate/preds')
    parser.add_argument('--hr', type=str, default='/home/mark/DataBank/PROBA-V-CHKPT/trimmedArrayDir')
    parser.add_argument('--norm', type=str, default='/home/mark/DataBank/probav_data/norm.csv')
    opt = parser.parse_args()
    return opt


def main():
    predImgFname = os.listdir(opt.preds)
    patchSize = 96

    allImg = loadHRImages()

    allImgMsk = generatePatchesPerImgSet(allImg, patchSize, patchSize)
    del allImg

    currBest = loadImagesIntoArray(
        '/home/mark/DataBank/PROBA-V-CHKPT/old/results/testout_patch38_top9_85p_12res_L1Loss')
    currBest = generatePatches(currBest, patchSize, patchSize)
    toCompare = loadImagesIntoArray('/home/mark/PROBA-V/trainout_16_top9_90p_8Res_32_L1Loss_postlearn')
    toCompare = generatePatches(toCompare, patchSize, patchSize)

    currBest = currBest.transpose((0, 2, 3, 1))
    toCompare = toCompare.transpose((0, 2, 3, 1))
    allImgMsk = allImgMsk.transpose((0, 2, 3, 1))
    currPSNR, compPSNR = calcRelativePSNR(currBest, toCompare, allImgMsk)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(currPSNR, compPSNR, edgecolors='k', alpha=0.6)

    ax.set_xlim([30, 70])
    ax.set_ylim([30, 70])
    ax.plot([30, 70], [30, 70], 'red', zorder=1)
    ax.set_xlabel('cPSNR: 32x32 12Res85Clarity')
    ax.set_ylabel('cPSNR: 16x16 8Res90Clarity')
    fig.show()


def calcRelativePSNR(patchPredOne, patchPredTwo, patchHR):
    patchSize = patchPredOne.shape[2]
    loss = Losses(targetShape=(patchSize, patchSize, 1))

    patchPredOne = tf.convert_to_tensor(patchPredOne, dtype=tf.float32)
    patchPredTwo = tf.convert_to_tensor(patchPredTwo, dtype=tf.float32)
    patchHRMask = tf.convert_to_tensor(~patchHR.mask, dtype=tf.float32)
    patchHR = tf.convert_to_tensor(patchHR, dtype=tf.float32)

    condOnePSNR = loss.shiftCompensatedcPSNR(patchHR, patchHRMask, patchPredOne)
    condTwoPSNR = loss.shiftCompensatedcPSNR(patchHR, patchHRMask, patchPredTwo)
    return condOnePSNR.numpy(), condTwoPSNR.numpy()


def loadImagesIntoArray(path):
    names = os.listdir(path)
    names = sorted(names)
    imgs = []
    for i, name in tqdm(enumerate(names), total=1160):
        if i == 1160:
            break
        img = io.imread(os.path.join(path, name))
        img = np.expand_dims(img, axis=0)
        imgs.append(img)
    imgs = np.concatenate(imgs)
    imgs = np.expand_dims(imgs, axis=1)
    imgs = imgs.astype(np.float32)
    return imgs


def loadHRImages():
    dirName = '/home/mark/DataBank/PROBA-V-CHKPT/trimmedArrayDir'
    red = 'TRAINimgHR_RED.npy'
    nir = 'TRAINimgHR_NIR.npy'

    red = np.load(os.path.join(dirName, red), allow_pickle=True)
    nir = np.load(os.path.join(dirName, nir), allow_pickle=True)
    allImg = np.ma.concatenate((red, nir))
    allImg = allImg.squeeze(1)
    allImg = allImg.astype(np.float32)
    return allImg


def generatePatches(images: np.array, patchSize: int, stride: int) -> np.array:
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

    numMskPerImgSet, channels, height, width = images.shape

    patchesImg = tensorImg.unfold(0, numMskPerImgSet, numMskPerImgSet).unfold(
        1, channels, channels).unfold(2, patchSize, stride).unfold(3, patchSize, stride)
    patchesImg = patchesImg.reshape(-1, channels, patchSize, patchSize)  # [numImgPerImgSet * numPatches, C, H, W]
    patchesImg = patchesImg.numpy()
    return patchesImg


def bicubicMean(img: np.array, upscale: int, coef: float = -0.5):
    H, W, C = img.shape

    img = padding(img, H, W, C)
    # Create new image
    dH = math.floor(H*upscale)
    dW = math.floor(W*upscale)
    dst = np.zeros((dH, dW, 3))

    h = 1/upscale

    print('Start bicubic interpolation')
    print('It will take a little while...')
    inc = 0
    for c in trange(C, desc='Channel loop'):
        for j in trange(dH, desc='Height loop', leave=False):
            for i in trange(dW, desc='Width loop', leave=False):
                x, y = i * h + 2, j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                mat_l = np.matrix([[u(x1, coef), u(x2, coef), u(x3, coef), u(x4, coef)]])
                mat_m = np.matrix([[img[int(y-y1), int(x-x1), c], img[int(y-y2), int(x-x1), c], img[int(y+y3), int(x-x1), c], img[int(y+y4), int(x-x1), c]],
                                   [img[int(y-y1), int(x-x2), c], img[int(y-y2), int(x-x2), c],
                                    img[int(y+y3), int(x-x2), c], img[int(y+y4), int(x-x2), c]],
                                   [img[int(y-y1), int(x+x3), c], img[int(y-y2), int(x+x3), c],
                                    img[int(y+y3), int(x+x3), c], img[int(y+y4), int(x+x3), c]],
                                   [img[int(y-y1), int(x+x4), c], img[int(y-y2), int(x+x4), c], img[int(y+y3), int(x+x4), c], img[int(y+y4), int(x+x4), c]]])
                mat_r = np.matrix([[u(y1, coef)], [u(y2, coef)], [u(y3, coef)], [u(y4, coef)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

    return dst


def padding(img, H, W, C):
    zimg = np.zeros((H+4, W+4, C))
    zimg[2:H+2, 2:W+2, :C] = img
    # Pad the first/last two col and row
    zimg[2:H+2, 0:2, :C] = img[:, 0:1, :C]
    zimg[H+2:H+4, 2:W+2, :] = img[H-1:H, :, :]
    zimg[2:H+2, W+2:W+4, :] = img[:, W-1:W, :]
    zimg[0:2, 2:W+2, :C] = img[0:1, :, :C]
    # Pad the missing eight points
    zimg[0:2, 0:2, :C] = img[0, 0, :C]
    zimg[H+2:H+4, 0:2, :C] = img[H-1, 0, :C]
    zimg[H+2:H+4, W+2:W+4, :C] = img[H-1, W-1, :C]
    zimg[0:2, W+2:W+4, :C] = img[0, W-1, :C]
    return zimg
