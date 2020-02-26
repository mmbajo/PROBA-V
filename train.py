from typing import List, Dict, Tuple

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD, Nadam

from modelsTF import *
from utils.loss import *
from utils.utils import *

import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('__name__')


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathchLR', type=str)
    parser.add_argument('--pathchHR', type=str)
    parser.add_argument('--split', type=float, default=0.7)
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--logDir', type=str, default='logs')
    parser.add_argument('--ckptDir', type=str, default='checkpoint')
    parser.add_argument('--optim', type=str, default='adam')
    opt = parser.parse_args()
    return opt


def main():
    # Safety checks
    if not os.path.exists(opt.ckptDir):
        os.makedirs(opt.ckptDir)
    if not os.path.exists(opt.logs):
        os.makedirs(opt.logs)

    logger.info('Building model...')
    model = WDSRConv3D(scale=3, numFilters=32, kernelSize=(3, 3, 3), numResBlocks=8,
                       expRate=8, decayRate=0.8, numImgLR=9, patchSizeLR=32)

    if opt.optimizer == 'adam':
        optimizer = Adam(learning_rate=5e-4)
    elif opt.optimizer == 'nadam':
        # http://cs229.stanford.edu/proj2015/054_report.pdf
        optimizer = Nadam(learning_rate=5e-4)
    else:
        optimizer = SGD(learning_rate=5e-4)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                     psnr=tf.Variable(1.0),
                                     optimizer=optimizer,
                                     model=model)

    checkpointManager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                   directory=opt.ckptDir,
                                                   max_to_keep=5)

    # Load Data
    logger.info('Loading data...')
    patchLR = np.load(opt.patchLR, allow_pickle=True)
    patchHR = np.load(opt.patchHR, allow_pickle=True)

    X_train, X_val, y_train, y_val, y_train_mask, y_val_mask = train_test_split(
        patchLR, patchHR, patchHR.mask, test_size=opt.split, random_state=17)

    y = [y_train, y_train_mask]
    valData = [X_val, [y_val, y_val_mask]]

    # Initialize metrics
    trainLoss = Mean(name='trainLoss')
    trainPSNR = Mean(name='trainPSNR')
    testLoss = Mean(name='testLoss')
    testPSNR = Mean(name='testPSNR')

    fitTrainData(model, optimizer, trainLoss, trainPSNR, testLoss, testPSNR,
                 X_train, y, opt.batchSize, opt.epochs, 512, valData, 100,
                 checkpoint, checkpointManager,
                 opt.logDir, opt.ckptDir, opt.saveBestOnly)


def fitTrainData(model, optimizer, trainLoss, trainPSNR, testLoss, testPSNR,
                 X, y, batchSize, epochs, bufferSize, valData, valSteps,
                 checkpoint, checkpointManager,
                 logDir, ckptDir, saveBestOnly):

    trainSet = loadTrainDataAsTFDataSet(X, y, epochs, batchSize, bufferSize)
    valSet = loadValDataAsTFDataSet(valData[0], valData[1], valSteps, batchSize, bufferSize)

    # Logger
    w = tf.summary.create_file_writer(logDir)

    dataSetLength = (*X, *y)[0].shape[0]
    totalSteps = tf.cast(dataSetLength/batchSize, tf.int64)
    globalStep = checkpoint.step
    step = globalStep % totalSteps
    epoch = 0

    with w.as_default():
        for x_batch_train, y_batch_train, y_mask_batch_train in trainSet:
            if (totalSteps - step) == 0:
                epoch += 1
                step = globalStep % totalSteps
                logger.info('Start of epoch %d' % (epoch))
                # Reset metrics
                trainLoss.reset_states()
                trainPSNR.reset_states()
                testLoss.reset_states()
                testPSNR.reset_states()

            step += 1
            globalStep += 1
            trainStep(x_batch_train, y_batch_train, y_mask_batch_train)
            checkpoint.step.assign_add(1)

            t = f"step {step}/{int(totalSteps)}, loss: {trainLoss.result():.3f}, psnr: {trainPSNR.result():.3f}"
            logger.info(t)

            tf.summary.scalar('Train PSNR', trainPSNR.result(), step=globalStep)

            tf.summary.scalar('Train loss', trainLoss.result(), step=globalStep)

            if step != 0 and (step % 100) == 0:
                # Reset states for test
                testLoss.reset_states()
                testPSNR.reset_states()
                for x_batch_val, y_batch_val, y_mask_batch_val in valSet:
                    testStep(x_batch_val, y_batch_val, y_mask_batch_val, checkpoint)
                tf.summary.scalar(
                    'Test loss', testLoss.result(), step=globalStep)
                tf.summary.scalar(
                    'Test PSNR', testPSNR.result(), step=globalStep)
                t = f"Validation results... val_loss: {testLoss.result():.3f}, val_psnr: {testPSNR.result():.3f}"
                logger.info(t)
                w.flush()

                if saveBestOnly and (testPSNR.result() <= checkpoint.psnr):
                    continue

                checkpoint.psnr = testPSNR.result()
                checkpointManager.save()


@tf.function
def trainStep(lr, mean_lr, hr, mask, checkpoint, loss, metric, trainLoss, trainPSNR):
    with tf.GradientTape() as tape:

        sr = checkpoint.model([lr, mean_lr], training=True)
        loss = loss(hr, sr, mask)

    gradients = tape.gradient(loss, checkpoint.model.trainable_variables)
    checkpoint.optimizer.apply_gradients(zip(gradients, checkpoint.model.trainable_variables))

    metric = metric(hr, mask, sr)
    trainLoss(loss)
    trainPSNR(metric)


@tf.function
def testStep(lr, mean_lr, hr, mask, checkpoint, loss, metric, testLoss, testPSNR):
    sr = checkpoint.model([lr, mean_lr], training=False)
    loss = loss(hr, mask, sr)
    metric = metric(hr, mask, sr)

    testLoss(loss)
    testPSNR(metric)


if __name__ == '__main__':
    opt = parser()
    main()
