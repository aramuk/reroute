##########################################################
# Repository of neural network layers used in the task.  #
##########################################################

import tensorflow as tf
import tensorflow.keras.layers as layers


class AdaIN(layers.Layer):
    """Adjusts the distribution of a content tensor to match a style tensor, instance-wise."""
    def __init__(self, epsilon=1e-6, name='adain', **kwargs):
        """Create an Adaptive Instance Normalization (AdaIN) layer."""
        super(AdaIN, self).__init__(name=name, **kwargs)
        self.epsilon = epsilon

    def call(self, x):
        """Performs adaptive IN on input content and style tensors
        
        Args:
        x: An input tensor representing a (content, style) pairing.

        Returns:
        The content tensor, instance normalized to match the distribution of the style tensor.
        """
        content, style = x
        mean_C, var_C = tf.nn.moments(content, axis=[1, 2], keepdims=True)
        mean_S, var_S = tf.nn.moments(style, axis=[1, 2], keepdims=True)
        std_C = tf.math.sqrt(var_C + self.epsilon)
        std_S = tf.math.sqrt(var_S + self.epsilon)
        return std_S * (content - mean_C) / std_C + mean_S


class ReflectionPad2D(layers.Layer):
    """Pads the height and width dimensions of an input tensor using reflection padding."""
    def __init__(self, padding=[[1, 1], [1, 1]], name='reflection', **kwargs):
        """Create a ReflectionPad2D layer with a given padding amount."""
        super(ReflectionPad2D, self).__init__(name=name, **kwargs)
        self.padding = padding

    def compute_output_shape(self, input_shape):
        """Returns the shape of a padded input."""
        return (input_shape[0], self.pad_l + input_shape[1] + self.pad_r,
                self.pad_t + input_shape[2] + self.pad_b, self.input_shape[3])

    def call(self, x):
        """Reflection pads an image."""
        return tf.pad(x, [[0, 0], *self.padding, [0, 0]], mode='reflect')