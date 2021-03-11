##########################################################
# Loss functions for style transfer.                     #
##########################################################

import tensorflow as tf


def _mse_loss(y_true, y_pred):
    """Calculates the MSE loss between two tensors."""
    tf.math.reduce_mean(tf.square(y_pred - y_true), axis=1)


def content_loss(y_true, y_pred):
    """Calculates loss between two content tensors using MSE."""
    return _mse_loss(y_true, y_pred)


def style_loss(y_true, y_pred):
    """Calculates loss between two stacks of style tensors using MSE of their mean and std."""
    return tf.sum(
        _mse_loss(tf.reduce_mean(phi_true, axis=(
            1, 2)), tf.reduce_mean(phi_pred, axis=(1, 2))) +
        _mse_loss(tf.reduce_std(phi_true, axis=(
            1, 2)), tf.reduce_std(phi_pred, axis=(1, 2)))
        for phi_true, phi_pred in zip(y_true, y_pred))
