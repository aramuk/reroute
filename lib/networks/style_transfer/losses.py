##########################################################
# Loss functions for style transfer.                     #
##########################################################

import tensorflow as tf


def _mse_loss(y_true, y_pred):
    """Calculates the MSE loss between two tensors."""
    return tf.math.reduce_mean(tf.square(y_pred - y_true))


def content_loss(y_true, y_pred):
    """Calculates loss between two content tensors using MSE."""
    return _mse_loss(y_true, y_pred)


def style_loss(y_true, y_pred):
    """Calculates loss between two stacks of style tensors using MSE of their mean and std."""
    return tf.reduce_sum([
        _mse_loss(tf.math.reduce_mean(phi_true, axis=(
            1, 2)), tf.math.reduce_mean(phi_pred, axis=(1, 2))) +
        _mse_loss(tf.math.reduce_std(phi_true, axis=(
            1, 2)), tf.math.reduce_std(phi_pred, axis=(1, 2)))
        for phi_true, phi_pred in zip(y_true, y_pred)])

if __name__ == '__main__':
    y_true = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.dtypes.float32)
    y_pred = tf.ones((3, 3), dtype=tf.dtypes.float32)

    print(_mse_loss(y_true, y_pred))