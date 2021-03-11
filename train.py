##########################################################
# Trains a style_transfer model.                         #
##########################################################

import os
import time

import tensorflow as tf
from tensorflow.keras import optimizers

from lib.networks.style_transfer.layers import TVLoss

class Trainer():
    def __init__(self,
                 model,
                 loss_fn=TVLoss(),
                 optimizer=optimizers.Adam,
                 learning_rate=1e-4,
                 lr_decay=5e-5,
                 momentum=0.9):
        self.model = model
        self.encoder = model.get_layer('mobilenetv2_encoder')
        self.loss_fn = loss_fn
        schedule = optimizers.schedules.InverseTimeDecay(learning_rate, 1, lr_decay)
        self.optimizer = optimizer(learning_rate=schedule, beta_1=momentum)

    @tf.function
    def _train_one_step(self, X, step, total_steps, epoch, total_epochs):
        with tf.GradientTape() as tape:
            output, target, _, style_features = self.model(X, training=True)
            output_features = self.encoder(output)
            loss = self.loss_fn([style_features, output_features, target])
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_weights))
        print('Epoch[{}/{}]; Step[{}/{}]; Loss = {}'.format(epoch, total_epochs, step, total_steps, loss))
        return loss

    def train(self, dataset, start=1, num_epochs=25):
        """Trains the style transfer model."""
        start = time.time()
        for epoch in range(start, start + num_epochs + 1):
            for step, X in enumerate(dataset):
                loss = self._train_one_step(X, step, len(dataset), epoch, start + num_epochs + 1)
            print('Epoch {} completed in {} seconds'.format(epoch, start - time.time()))
            self.save_model('./models/', tag='_checkpoint', overwrite=True)

    def save_model(self, save_dir, tag='_weights', overwrite=False):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        file_name = os.path.join(save_dir, '{}_{}.h5'.format(self.model.name, tag))
        if os.path.exists(file_name):
            if overwrite:
                os.remove(file_name)
            else:
                raise ValueError('File at {} already exists'.format(file_name))
        self.model.save_weights(file_name)

    def load_model(self, file_path):
        if not os.path.exists(file_path):
            raise ValueError('Could not find a weights file at {}'.format(file_path))
        self.model.load_weights(file_path)

if __name__ == '__main__':
    from lib.networks.style_transfer.network import build_model
    MobileAdaIN = build_model()
    trainer = Trainer(MobileAdaIN)
