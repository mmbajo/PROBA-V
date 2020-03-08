from typing import List

import os
import tensorflow as tf
import numpy as np

from utils.utils import loadTrainDataAsTFDataSet, loadValDataAsTFDataSet
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

    def __init__(self, model, loss, metric, optimizer, ckptDir, logDir, multiGPU=True, evalStep=1000):

        # Safety checks
        self.logDirTrain = os.path.join(logDir, 'Train')
        self.logDirTest = os.path.join(logDir, 'Test')

        if not os.path.exists(ckptDir):
            os.makedirs(ckptDir)
        if not os.path.exists(self.logDirTrain):
            os.makedirs(self.logDirTrain)
        if not os.path.exists(self.logDirTest):
            os.makedirs(self.logDirTest)

        self.trainWriter = tf.summary.create_file_writer(self.logDirTrain)
        self.testWriter = tf.summary.create_file_writer(self.logDirTest)

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                                        psnr=tf.Variable(1.0),
                                        optimizer=optimizer,
                                        model=model)
        self.ckptMngr = tf.train.CheckpointManager(checkpoint=self.ckpt,
                                                   directory=ckptDir,
                                                   max_to_keep=5)

        self.loss = loss
        self.metric = metric

        self.accTestLoss = Mean(name='accTestLoss')
        self.accTestPSNR = Mean(name='accTestPSNR')
        self.accTrainLoss = Mean(name='accTrainLoss')
        self.accTrainPSNR = Mean(name='accTrainPSNR')
        self.evalStep = evalStep
        self.multiGPU = multiGPU
        self.strategy = None
        self.restore()

    @property
    def model(self):
        return self.ckpt.model

    def restore(self):
        if self.ckptMngr.latest_checkpoint:
            self.ckpt.restore(self.ckptMngr.latest_checkpoint)
            print(f'[ INFO ] Model restored from checkpoint at step {self.ckpt.step.numpy()}.')

    def fitTrainData(self,
                     X: tf.Tensor, y: tf.Tensor,
                     globalBatchSize: int, epochs: int,
                     valData: List[np.ma.array],
                     bufferSize: int = 128, valSteps: int = 64,
                     saveBestOnly: bool = True, initEpoch: int = 0):

        logger.info('[ INFO ] Loading data set to buffer cache...')
        trainSet = loadTrainDataAsTFDataSet(X, y[0], y[1], epochs, globalBatchSize, bufferSize)
        valSet = loadValDataAsTFDataSet(valData[0], valData[1], valData[2], valSteps, globalBatchSize, bufferSize)
        logger.info('[ INFO ] Loading success...')

        dataSetLength = len(X)
        totalSteps = tf.cast(dataSetLength/globalBatchSize, tf.int64)
        globalStep = tf.cast(self.ckpt.step, tf.int64)
        step = globalStep % totalSteps
        epoch = initEpoch

        logger.info('[ INFO ] Begin training...')

        for x_batch_train, y_batch_train, y_mask_batch_train in trainSet:
            if (totalSteps - step) == 0:
                epoch += 1
                step = tf.cast(self.ckpt.step, tf.int64) % totalSteps
                logger.info(f'[ ***************  NEW EPOCH  *************** ] Epoch number {epoch}')
                # Reset metrics
                self.accTrainLoss.reset_states()
                self.accTrainPSNR.reset_states()
                self.accTestLoss.reset_states()
                self.accTestPSNR.reset_states()

            step += 1
            globalStep += 1
            self.trainStep(x_batch_train, y_batch_train, y_mask_batch_train)
            self.ckpt.step.assign_add(1)

            t = f"[ EPOCH {epoch}/{epochs} ] - [ STEP {step}/{int(totalSteps)} ] Loss: {self.accTrainLoss.result():.3f}, cPSNR: {self.accTrainPSNR.result():.3f}"
            logger.info(t)

            self.saveLog('Train', globalStep)

            if step != 0 and (step % self.evalStep) == 0:
                # Reset states for test
                self.accTestLoss.reset_states()
                self.accTestPSNR.reset_states()
                for x_batch_val, y_batch_val, y_mask_batch_val in valSet:
                    self.testStep(x_batch_val, y_batch_val, y_mask_batch_val)
                self.saveLog('Test', globalStep)
                t = f"[ *************** VAL INFO *************** ] Validation Loss: {self.accTestLoss.result():.3f}, Validation PSNR: {self.accTestPSNR.result():.3f}"
                logger.info(t)

                if saveBestOnly and (self.accTestPSNR.result() <= self.ckpt.psnr):
                    continue

                logger.info('[ SAVE ] Saving checkpoint...')
                self.ckpt.psnr = self.accTestPSNR.result()
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
        self.accTrainLoss(loss)
        self.accTrainPSNR(metric)

    @tf.function
    def testStep(self, patchLR, patchHR, maskHR):
        predPatchHR = self.ckpt.model(patchLR, training=False)
        loss = self.loss(patchHR, maskHR, predPatchHR)
        metric = self.metric(patchHR, maskHR, predPatchHR)
        self.accTestLoss(loss)
        self.accTestPSNR(metric)

    def saveLog(self, testOrTrain, globalStep):
        w = self.trainWriter if testOrTrain == 'Train' else self.testWriter
        with w.as_default():
            if testOrTrain == 'Train':
                tf.summary.scalar('PSNR', self.accTrainPSNR.result(), step=globalStep)
                tf.summary.scalar('Loss', self.accTrainLoss.result(), step=globalStep)
            else:
                tf.summary.scalar('PSNR', self.accTestPSNR.result(), step=globalStep)
                tf.summary.scalar('Loss', self.accTestLoss.result(), step=globalStep)
            w.flush()
