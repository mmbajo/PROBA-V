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

    def __init__(self, model, loss, metric, optimizer, ckptDir, logDir, config, multiGPU=True, evalStep=1000):

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
        self.multiGPU = multiGPU
        self.strategy = None
        self.batchSize = config['batch_size']
        self.epochs = config['epochs']
        self.lr = config['learning_rate']
        self.restore()

    @property
    def model(self):
        return self.ckpt.model

    def decay(self, epoch):
        if epoch < 10:
            return self.lr
        if epoch >= 10 and epoch < 20:
            return self.lr/2
        if epoch >= 20:
            return self.lr/10

    def restore(self):
        if self.ckptMngr.latest_checkpoint:
            self.ckpt.restore(self.ckptMngr.latest_checkpoint)
            print(f'[ INFO ] Model restored from checkpoint at step {self.ckpt.step.numpy()}.')

    def fitTrainData(self,
                     X: tf.Tensor, y: tf.Tensor,
                     valData: List[np.ma.array],
                     bufferSize: int = 256, valSteps: int = 64,
                     saveBestOnly: bool = False, initEpoch: int = 0):

        logger.info('[ INFO ] Loading data set to buffer cache...')
        trainSet = loadTrainDataAsTFDataSet(X, y[0], y[1], self.epochs, self.batchSize, bufferSize)
        valSet = loadValDataAsTFDataSet(valData[0], valData[1], valData[2], valSteps, self.batchSize, bufferSize)
        logger.info('[ INFO ] Loading success...')

        w = tf.summary.create_file_writer(self.logDir)

        dataSetLength = len(X)
        totalSteps = tf.cast(dataSetLength/self.batchSize, tf.int64)
        globalStep = tf.cast(self.ckpt.step, tf.int64)
        step = globalStep % totalSteps
        epoch = initEpoch

        logger.info('[ INFO ] Begin training...')
        with w.as_default():
            for x_batch_train, y_batch_train, y_mask_batch_train in trainSet:
                if (totalSteps - step) == 0:
                    epoch += 1
                    step = tf.cast(self.ckpt.step, tf.int64) % totalSteps
                    logger.info(f'[ ***************  NEW EPOCH  *************** ] Epoch number {epoch}')
                    # Reset metrics
                    self.trainLoss.reset_states()
                    self.trainPSNR.reset_states()
                    self.testLoss.reset_states()
                    self.testPSNR.reset_states()

                step += 1
                globalStep += 1
                self.trainStep(x_batch_train, y_batch_train, y_mask_batch_train)
                self.ckpt.step.assign_add(1)

                t = f"[ EPOCH {epoch}/{self.epochs} ] - [ STEP {step}/{int(totalSteps)} ] Loss: {self.trainLoss.result():.6f}, cPSNR: {self.trainPSNR.result():.3f}"
                logger.info(t)

                tf.summary.scalar('Train PSNR', self.trainPSNR.result(), step=globalStep)
                tf.summary.scalar('Train loss', self.trainLoss.result(), step=globalStep)

                if step != 0 and (step % self.evalStep) == 0:
                    # Reset states for test
                    self.testLoss.reset_states()
                    self.testPSNR.reset_states()
                    for x_batch_val, y_batch_val, y_mask_batch_val in valSet:
                        self.testStep(x_batch_val, y_batch_val, y_mask_batch_val)
                    tf.summary.scalar('Test loss', self.testLoss.result(), step=globalStep)
                    tf.summary.scalar('Test PSNR', self.testPSNR.result(), step=globalStep)
                    t = f"[ *************** VAL INFO *************** ] Validation Loss: {self.testLoss.result():.6f}, Validation PSNR: {self.testPSNR.result():.3f}"
                    logger.info(t)
                    w.flush()

                    if saveBestOnly and (self.testPSNR.result() <= self.ckpt.psnr):
                        continue

                    logger.info('[ SAVE ] Saving checkpoint...')
                    self.ckpt.psnr = self.testPSNR.result()
                    self.ckptMngr.save()

    def computeLoss(self, patchHR, maskHR, predPatchHR):
        loss = tf.reduce_sum(self.loss(patchHR, maskHR, predPatchHR)) * (1.0 / self.batchSize)
        loss += (sum(self.ckpt.model.losses) * 1.0 / self.strategy.num_replicas_in_sync)
        return loss

    def calcMetric(self, patchHR, maskHR, predPatchHR):
        metric = tf.reduce_sum(self.metric(patchHR, maskHR, predPatchHR)) * (1.0 / self.batchSize)
        metric *= (1.0 / self.strategy.num_replicas_in_sync)
        return metric

    @tf.function
    def trainStep(self, patchLR, patchHR, maskHR):
        with tf.GradientTape() as tape:
            predPatchHR = self.ckpt.model(patchLR, training=True)
            # Loss(patchHR: tf.Tensor, maskHR: tf.Tensor, predPatchHR: tf.Tensor)
            loss = self.computeLoss(patchHR, maskHR, predPatchHR)

        gradients = tape.gradient(loss, self.ckpt.model.trainable_variables)
        self.ckpt.optimizer.apply_gradients(zip(gradients, self.ckpt.model.trainable_variables))
        metric = self.calcMetric(patchHR, maskHR, predPatchHR)
        return loss, metric

    @tf.function
    def testStep(self, patchLR, patchHR, maskHR):
        predPatchHR = self.ckpt.model(patchLR, training=False)
        loss = self.computeLoss(patchHR, maskHR, predPatchHR)
        metric = self.calcMetric(patchHR, maskHR, predPatchHR)
        return loss, metric

    @tf.function
    def trainDistStep(self, patchLR, patchHR, maskHR):
        perExampleLosses, perExampleMetric = self.strategy.experimental_run_v2(
            self.trainStep, args=(patchLR, patchHR, maskHR))
        meanLoss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, perExampleLosses, axis=None)
        meanMetric = self.strategy.reduce(tf.distribute.ReduceOp.SUM, perExampleMetric, axis=None)
        self.trainLoss(meanLoss)
        self.trainPSNR(meanMetric)

    @tf.function
    def testDistStep(self, patchLR, patchHR, maskHR):
        perExampleLosses, perExampleMetric = self.strategy.experimental_run_v2(
            self.testStep, args=(patchLR, patchHR, maskHR))
        meanLoss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, perExampleLosses, axis=None)
        meanMetric = self.strategy.reduce(tf.distribute.ReduceOp.SUM, perExampleMetric, axis=None)
        self.testLoss(meanLoss)
        self.testPSNR(meanMetric)
