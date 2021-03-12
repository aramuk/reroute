##########################################################
# Trains a style_transfer model.                         #
##########################################################

import os
import time

import cv2
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from lib.networks.style_transfer.layers import TVLoss


class Trainer():

    def __init__(self,
                 model,
                 loss_fn=TVLoss(),
                 optimizer=optimizers.Adam,
                 learning_rate=1e-4,
                 lr_decay=5e-5,
                 momentum=0.9,
                 log_dir='./logs',
                 save_dir='./models'):
        self.model = model
        self.encoder = model.get_layer('mobilenetv2_encoder')
        self.loss_fn = loss_fn
        schedule = optimizers.schedules.InverseTimeDecay(
            learning_rate, 1, lr_decay)
        self.optimizer = optimizer(learning_rate=schedule, beta_1=momentum)
        self.tensorboard = callbacks.TensorBoard(log_dir=log_dir,
                                                 write_graph=True)
        self.tensorboard.set_model(self.model)
        self.save_dir = save_dir
        self.reset_epochs()

    def reset_epochs(self):
        self.epoch = 1
        self.total_epochs = 0

    def _train_one_step(self,
                        X,
                        step,
                        total_steps,
                        output_freq=100):
        with tf.GradientTape() as tape:
            output, target, _, style_features = self.model(X, training=True)
            # if step == 1:
            #     cv2.imwrite('./output_' + str(self.epoch) + '_0.png', output.numpy()[0]*255)
            output_features = self.encoder(output)
            loss = self.loss_fn([style_features, output_features, target])
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.tensorboard.on_batch_end(step, {'loss': loss})
        loss = loss.numpy().item()
        if step % output_freq == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'.format(
                self.epoch, self.total_epochs, step, total_steps, loss))
        return loss

    def train(self, dataset, num_epochs=25, output_freq=100):
        """Trains the style transfer model."""
        self.total_epochs += num_epochs
        start = time.time()
        while self.epoch <= self.total_epochs:
            epoch_loss = 0.0
            for step, X in enumerate(dataset, start=1):
                loss = self._train_one_step(X, step, len(dataset), output_freq)
                epoch_loss += loss
            print('------------[Epoch {}; Loss = {:.6f}]------------'.format(
                self.epoch, epoch_loss))
            self.epoch += 1
            self.tensorboard.on_epoch_end(step, {'loss': loss})
            self.save_model(tag='_checkpoint', overwrite=True)
        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    def save_model(self, tag='_weights', overwrite=False):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        file_name = os.path.join(self.save_dir,
                                 '{}_{}.h5'.format(self.model.name, tag))
        if os.path.exists(file_name):
            if overwrite:
                os.remove(file_name)
            else:
                raise ValueError('File at {} already exists'.format(file_name))
        self.model.save_weights(file_name)

    def load_model(self, file_path):
        if not os.path.exists(file_path):
            raise ValueError(
                'Could not find a weights file at {}'.format(file_path))
        self.model.load_weights(file_path)
