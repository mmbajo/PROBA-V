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

    def __init__(self, model, loss, metric, optimizer, ckptDir, logDir, strategy, multiGPU=True, evalStep=10):

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
                                                   max_to_keep=20)
        self.loss = loss
        self.metric = metric
        self.logDir = logDir
        self.trainLoss = Mean(name='trainLoss')
        self.trainPSNR = Mean(name='trainPSNR')
        self.testLoss = Mean(name='testLoss')
        self.testPSNR = Mean(name='testPSNR')
        self.evalStep = evalStep
        self.multiGPU = multiGPU
        self.strategy = strategy
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
                     batchSize: int, epochs: int,
                     valData: List[np.ma.array],
                     bufferSize: int = 256, valSteps: int = 128,
                     saveBestOnly: bool = True, initEpoch: int = 0):
        if self.multiGPU:
            logger.info('[ INFO ] Multi-GPU mode selected...')
            logger.info('[ INFO ] Instantiate strategy...')
            batchSizePerReplica = batchSize
            globalBatchSize = batchSizePerReplica * self.strategy.num_replicas_in_sync
        else:
            globalBatchSize = batchSize

        logger.info('[ INFO ] Loading data set to buffer cache...')
        trainSet = loadTrainDataAsTFDataSet(X, y[0], y[1], epochs, globalBatchSize, bufferSize)
        valSet = loadValDataAsTFDataSet(valData[0], valData[1], valData[2], valSteps, globalBatchSize, bufferSize)
        logger.info('[ INFO ] Loading success...')

        if self.multiGPU:
            logger.info('[ INFO ] Distributing train set...')
            trainSet = self.strategy.experimental_distribute_dataset(trainSet)
            logger.info('[ INFO ] Distributing test set...')
            valSet = self.strategy.experimental_distribute_dataset(valSet)

        w = tf.summary.create_file_writer(self.logDir)

        dataSetLength = len(X)
        totalSteps = tf.cast(dataSetLength/globalBatchSize, tf.int64)
        globalStep = tf.cast(self.ckpt.step, tf.int64)
        step = globalStep % totalSteps
        epoch = initEpoch

        logger.info('[ INFO ] Begin training...')
        with w.as_default():
            for x_batch_train, y_batch_train, y_mask_batch_train in trainSet:
                if (totalSteps - step) == 0:
                    epoch += 1
                    step = tf.cast(self.ckpt.step, tf.int64) % totalSteps
                    logger.info(f'[ NEW EPOCH ] Epoch number {epoch}')
                    # Reset metrics
                    self.trainLoss.reset_states()
                    self.trainPSNR.reset_states()
                    self.testLoss.reset_states()
                    self.testPSNR.reset_states()

                step += 1
                globalStep += 1
                self.trainDistStep(x_batch_train, y_batch_train, y_mask_batch_train)
                self.ckpt.step.assign_add(1)

                t = f"[ EPOCH {epoch}/{epochs} ] Step {step}/{int(totalSteps)}, Loss: {self.trainLoss.result():.3f}, cPSNR: {self.trainPSNR.result():.3f}"
                logger.info(t)

                tf.summary.scalar('Train PSNR', self.trainPSNR.result(), step=globalStep)
                tf.summary.scalar('Train loss', self.trainLoss.result(), step=globalStep)

                if step != 0 and (step % self.evalStep) == 0:
                    # Reset states for test
                    self.testLoss.reset_states()
                    self.testPSNR.reset_states()
                    for x_batch_val, y_batch_val, y_mask_batch_val in valSet:
                        self.testDistStep(x_batch_val, y_batch_val, y_mask_batch_val)
                    tf.summary.scalar('Test loss', self.testLoss.result(), step=globalStep)
                    tf.summary.scalar('Test PSNR', self.testPSNR.result(), step=globalStep)
                    t = f"[ VAL INFO ] Validation Loss: {self.testLoss.result():.3f}, Validation PSNR: {self.testPSNR.result():.3f}"
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

    @tf.function
    def trainStep(self, patchLR, patchHR, maskHR):
        with tf.GradientTape() as tape:
            predPatchHR = self.ckpt.model(patchLR, training=True)
            # Loss(patchHR: tf.Tensor, maskHR: tf.Tensor, predPatchHR: tf.Tensor)
            loss = self.loss(patchHR, maskHR, predPatchHR)

        gradients = tape.gradient(loss, self.ckpt.model.trainable_variables)
        self.ckpt.optimizer.apply_gradients(zip(gradients, self.ckpt.model.trainable_variables))
        return loss

    @tf.function
    def testStep(self, patchLR, patchHR, maskHR):
        predPatchHR = self.ckpt.model(patchLR, training=False)
        loss = self.loss(patchHR, maskHR, predPatchHR)
        return loss

    @tf.function
    def calcMetric(self, patchLR, patchHR, maskHR):
        return self.metric(patchHR, maskHR, predPatchHR)

    @tf.function
    def trainDistStep(self, patchLR, patchHR, maskHR):
        perExampleLosses = self.strategy.experimental_run_v2(self.trainStep, args=(patchLR, patchHR, maskHR))
        perExampleMetric = self.strategy.experimental_run_v2(self.calcMetric, args=(patchLR, patchHR, maskHR))
        meanLoss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, perExampleLosses, axis=0)
        meanMetric = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, perExampleMetric, axis=0)
        self.trainLoss(meanLoss)
        self.trainPSNR(meanMetric)

    @tf.function
    def testDistStep(self, patchLR, patchHR, maskHR):
        perExampleLosses = self.strategy.experimental_run_v2(self.testStep, args=(patchLR, patchHR, maskHR))
        perExampleMetric = self.strategy.experimental_run_v2(self.calcMetric, args=(patchLR, patchHR, maskHR))
        meanLoss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, perExampleLosses, axis=0)
        meanMetric = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, perExampleMetric, axis=0)
        self.testLoss(meanLoss)
        self.testPSNR(meanMetric)
