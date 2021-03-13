##########################################################
# Main training loop.                                    #
##########################################################

import argparse

import tensorflow as tf
from tensorflow.keras import optimizers

from lib.dataloader import StyleTransferDataLoader
from lib.networks.style_transfer.layers import TVLoss
from lib.networks.style_transfer.network import build_model
from lib.networks.style_transfer.trainer import Trainer


def train_model(content_height, content_width, style_height, style_width,
                crop_dim, save_dir, log_dir, optimizer, learning_rate, lr_decay,
                momentum, batch_size, num_epochs, output_freq):
    train_loader = StyleTransferDataLoader(
        batch_size=batch_size,
        length=280,
        content_transform=lambda I: tf.image.random_crop(
            tf.image.resize_with_pad(I, target_height=content_height, target_width=content_width),
            (crop_dim, crop_dim, 3)),
        style_transform=lambda I: tf.image.random_crop(
            tf.image.resize_with_crop_or_pad(I, target_height=style_height, target_width=style_width),
            (crop_dim, crop_dim, 3)))

    MobileAdaIN = build_model(input_shape=(crop_dim, crop_dim, 3))
    trainer = Trainer(
        MobileAdaIN,
        loss_fn=TVLoss(),
        optimizer=optimizers.Adam if optimizer == 'adam' else optimizers.SGD,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        momentum=momentum,
        log_dir=log_dir,
        save_dir=save_dir)

    trainer.train(train_loader, num_epochs=num_epochs, output_freq=output_freq)


if __name__ == '__main__':
    # yapf: disable
    parser = argparse.ArgumentParser(description='Train the ReRoute stylee transfer model')
    # Preprocessing
    parser.add_argument('--content_height', type=int, default=256, help='The resized height of the content images')
    parser.add_argument('--content_width', type=int, default=512, help='The resized width of the content images')
    parser.add_argument('--style_height', type=int, default=256, help='The resized height of the style images')
    parser.add_argument('--style_width', type=int, default=256, help='The resized width of the style images')
    parser.add_argument('--crop_dim', type=int, default=224, help='The dimensions of the random crop')
    # Logs
    parser.add_argument('--save_dir', default='./models', help='The directory to save the model weights at checkpoints')
    parser.add_argument('--log_dir', default='./models', help='The directory to save TensorBoard logs')
    # Training
    parser.add_argument('--optimizer', default='adam', help='The optimization algorithm to apply to the decoder')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='The initial learning rate for the optimizer')
    parser.add_argument('--lr_decay', type=float, default=5e-5, help='The rate of decay for the learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='The momentum for the optimizer')
    parser.add_argument('--batch_size', type=int, default=8, help='The number of training samples in a batch')
    parser.add_argument('--num_epochs', type=int, default=25, help='The number of training epochs')
    parser.add_argument('--output_freq', type=int, default=100, help='How frequently to output batch information')
    # yapf: enable
    args = parser.parse_args()
    train_model(**vars(args))