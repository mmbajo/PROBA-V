from typing import List

import os
import tensorflow as tf
import numpy as np

#from utils.utils import *
from tensorflow.keras.metrics import Mean

import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logger = logging.getLogger('__name__')


class ModelTrainer:
    """
    Note:
    Having this model keeps the trainStep and testStep instance new every time you call it.
    Implementing those functions outside a class will return an error
    ValueError: Creating variables on a non-first call to a function decorated with tf.function.
    """

    def __init__(self, model, loss, metric, optimizer, ckptDir, logDir, evalStep=100):

        # Safety checks
        if not os.path.exists(ckptDir):
            os.makedirs(ckptDir)
        if not os.path.exists(logDir):
            os.makedirs(logDir)

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                                        psnr=tf.Variable(1.0),
                                        optimizer=optimizer,
                                        model=model)
        self.ckptMngr = tf.train.CheckpointManager(checkpoint=self.ckpt,
                                                   directory=ckptDir,
                                                   max_to_keep=5)
        self.loss = loss
        self.metric = metric
        self.logDir = logDir
        self.trainLoss = Mean(name='trainLoss')
        self.trainPSNR = Mean(name='trainPSNR')
        self.testLoss = Mean(name='testLoss')
        self.testPSNR = Mean(name='testPSNR')
        self.evalStep = evalStep
        self.restore()

    @property
    def model(self):
        return self.ckpt.model

    def restore(self):
        if self.ckptMngr.latest_checkpoint:
            self.ckpt.restore(self.ckptMngr.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.ckpt.step.numpy()}.')

    def fitTrainData(self,
                     X: np.ma.array, y: np.ma.array,
                     batchSize: int, epochs: int, bufferSize: int,
                     valData: List[np.ma.array], valSteps: int,
                     saveBestOnly: bool = True, initEpoch: int = 0):

        trainSet = loadTrainDataAsTFDataSet(X, y[0], y[1], epochs, batchSize, bufferSize)
        valSet = loadValDataAsTFDataSet(valData[0], valData[1], valData[2], valSteps, batchSize, bufferSize)
        # Logger
        w = tf.summary.create_file_writer(self.logDir)

        dataSetLength = len(X)
        totalSteps = tf.cast(dataSetLength/batchSize, tf.int64)
        globalStep = tf.cast(self.ckpt.step, tf.int64)
        step = globalStep % totalSteps
        epoch = initEpoch

        with w.as_default():
            for x_batch_train, y_batch_train, y_mask_batch_train in trainSet:
                if (totalSteps - step) == 0:
                    epoch += 1
                    step = tf.cast(self.ckpt.step, tf.int64) % totalSteps
                    logger.info('Start of epoch %d' % (epoch))
                    # Reset metrics
                    self.trainLoss.reset_states()
                    self.trainPSNR.reset_states()
                    self.testLoss.reset_states()
                    self.testPSNR.reset_states()

                step += 1
                globalStep += 1
                self.trainStep(x_batch_train, y_batch_train, y_mask_batch_train)
                self.ckpt.step.assign_add(1)

                t = f"step {step}/{int(totalSteps)}, loss: {self.trainLoss.result():.3f}, psnr: {self.trainPSNR.result():.3f}"
                logger.info(t)

                tf.summary.scalar('Train PSNR', self.trainPSNR.result(), step=globalStep)
                tf.summary.scalar('Train loss', self.trainLoss.result(), step=globalStep)

                if step != 0 and (step % self.evalStep) == 0:
                    # Reset states for test
                    self.testLoss.reset_states()
                    self.testPSNR.reset_states()
                    for x_batch_val, y_batch_val, y_mask_batch_val in valSet:
                        self.testStep(x_batch_val, y_batch_val, y_mask_batch_val)
                    tf.summary.scalar(
                        'Test loss', self.testLoss.result(), step=globalStep)
                    tf.summary.scalar(
                        'Test PSNR', self.testPSNR.result(), step=globalStep)
                    t = f"Validation results... val_loss: {self.testLoss.result():.3f}, val_psnr: {self.testPSNR.result():.3f}"
                    logger.info(t)
                    w.flush()

                    if saveBestOnly and (self.testPSNR.result() <= self.ckpt.psnr):
                        continue

                    self.ckpt.psnr = self.testPSNR.result()
                    self.ckptMngr.save()

    @tf.function
    def trainStep(self, patchLR, patchHR, maskHR):
        with tf.GradientTape() as tape:
            predPatchHR = self.ckpt.model(patchLR, training=True)
            # Loss(patchHR: tf.Tensor, maskHR: tf.Tensor, predPatchHR: tf.Tensor)
            loss = self.loss(patchHR, maskHR, predPatchHR)

        gradients = tape.gradient(loss, self.ckpt.model.trainable_variables)
        self.ckpt.optimizer.apply_gradients(zip(gradients, self.ckpt.model.trainable_variables))

        metric = self.metric(patchHR, maskHR, predPatchHR)
        self.trainLoss(loss)
        self.trainPSNR(metric)

    @tf.function
    def testStep(self, patchLR, patchHR, maskHR):
        predPatchHR = self.ckpt.model(patchLR, training=False)
        loss = self.loss(patchHR, maskHR, predPatchHR)
        metric = self.metric(patchHR, maskHR, predPatchHR)

        self.testLoss(loss)
        self.testPSNR(metric)


def loadTrainDataAsTFDataSet(X, y, y_mask, epochs, batchSize, bufferSize):
    return tf.data.Dataset.from_tensor_slices(
        (X, y, y_mask)).shuffle(bufferSize, reshuffle_each_iteration=True).repeat(epochs).batch(batchSize).prefetch(tf.data.experimental.AUTOTUNE)


def loadValDataAsTFDataSet(X, y, y_mask, valSteps, batchSize, bufferSize):
    return tf.data.Dataset.from_tensor_slices(
        (X, y, y_mask)).shuffle(bufferSize).batch(batchSize).prefetch(tf.data.experimental.AUTOTUNE).take(valSteps)
