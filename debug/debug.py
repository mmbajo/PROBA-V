import tensorflow as tf
import time
import gc
from tensorflow.keras.metrics import Mean

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logger = logging.getLogger('__name__')


class Trainer:
    def __init__(self,
                 model,
                 loss, metric,
                 optimizer,
                 checkpoint_dir='./ckpt/3dsrnet', log_dir='logs'):

        self.now = None
        self.loss = loss
        self.metric = metric
        self.log_dir = log_dir
        self.train_loss = Mean(name='train_loss')
        self.train_psnr = Mean(name='train_psnr')

        self.test_loss = Mean(name='test_loss')
        self.test_psnr = Mean(name='test_psnr')
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(1.0),
                                              optimizer=optimizer,
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=5)

        self.restore()

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

    @property
    def model(self):
        return self.checkpoint.model
    
    def fit(self, x=None, y=None, batch_size=None, buffer_size=512, epochs=100,
            verbose=1, evaluate_every=100, val_steps=100,
            validation_data=None, shuffle=True, initial_epoch=0, save_best_only=True):

        ds_len = (*x, *y)[0].shape[0]
        # Create dataset from slices
        train_ds = tf.data.Dataset.from_tensor_slices(
            (*x, *y)).shuffle(buffer_size, reshuffle_each_iteration=True).repeat(epochs).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices(
            (*validation_data[0], *validation_data[1])).shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).take(val_steps)

        # Tensorboard logger
        writer = tf.summary.create_file_writer(self.log_dir)

        total_steps = tf.cast(ds_len/batch_size,tf.int64)
        global_step = tf.cast(self.checkpoint.step,tf.int64)
        step = tf.cast(self.checkpoint.step,tf.int64) % total_steps 
        epoch = initial_epoch
        with writer.as_default():


        # Iterate over the batches of the dataset.
            for x_batch_train, x_mean_batch_train, y_batch_train, y_mask_batch_train in train_ds:
                if (total_steps - step) == 0:
                    epoch+=1
                    step = tf.cast(self.checkpoint.step,tf.int64) % total_steps 
                    logger.info('Start of epoch %d' % (epoch))
                    # Reset metrics
                    self.train_loss.reset_states()
                    self.train_psnr.reset_states()
                    self.test_loss.reset_states()
                    self.test_psnr.reset_states()

                step +=1
                global_step += 1
                self.train_step(x_batch_train, x_mean_batch_train,
                                y_batch_train, y_mask_batch_train)
                self.checkpoint.step.assign_add(1)

                template = f"step {step}/{int(total_steps)}, loss: {self.train_loss.result():.3f}, psnr: {self.train_psnr.result():.3f}"
                logger.info(template)

                tf.summary.scalar(
                    'Train PSNR', self.train_psnr.result(), step=global_step)

                tf.summary.scalar(
                    'Train loss', self.train_loss.result(), step=global_step)

                if step != 0 and (step % evaluate_every) == 0:
                    # Reset states for test
                    self.test_loss.reset_states()
                    self.test_psnr.reset_states()
                    for x_batch_val, x_mean_batch_val, y_batch_val, y_mask_batch_val in val_ds:
                        self.test_step(
                            x_batch_val, x_mean_batch_val, y_batch_val, y_mask_batch_val)
                    tf.summary.scalar(
                        'Test loss', self.test_loss.result(), step=global_step)
                    tf.summary.scalar(
                        'Test PSNR', self.test_psnr.result(), step=global_step)
                    template = f"Validation results... val_loss: {self.test_loss.result():.3f}, val_psnr: {self.test_psnr.result():.3f}"
                    logger.info(template)
                    writer.flush()

                    if save_best_only and (self.test_psnr.result() <= self.checkpoint.psnr):
                        # skip saving checkpoint, no PSNR improvement
                        continue
                    self.checkpoint.psnr = self.test_psnr.result()
                    self.checkpoint_manager.save()

    @tf.function
    def train_step(self, lr, mean_lr, hr, mask):
        with tf.GradientTape() as tape:

            sr = self.checkpoint.model([lr, mean_lr], training=True)
            loss = self.loss(hr, sr, mask)

        gradients = tape.gradient(
            loss, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables))

        metric = self.metric(hr, sr, mask)
        self.train_loss(loss)
        self.train_psnr(metric)

    @tf.function
    def test_step(self, lr, mean_lr, hr, mask):
        sr = self.checkpoint.model([lr, mean_lr], training=False)
        t_loss = self.loss(hr, sr, mask)
        t_metric = self.metric(hr, sr, mask)

        self.test_loss(t_loss)
        self.test_psnr(t_metric)