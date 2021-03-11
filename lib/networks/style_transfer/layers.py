##########################################################
# Repository of neural network layers used in the task.  #
##########################################################

import tensorflow as tf
import tensorflow.keras.layers as layers

from .losses import content_loss, style_loss


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
        mean_C, var_C = tf.nn.moments(content, axes=[1, 2], keepdims=True)
        mean_S, var_S = tf.nn.moments(style, axes=[1, 2], keepdims=True)
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
        return tf.pad(x, [[0, 0], *self.padding, [0, 0]], mode='REFLECT')

    def get_config(self):
        """Get a dict of assigned options for this layer."""
        base_config = super(ReflectionPad2D, self).get_config()
        config = {'padding': self.padding, **base_config}
        return config


class DecoderBlock(layers.Layer):
    """A decoder block inpsired by MobileNetV2."""
    def __init__(self, filters, name='mnetv2_block', **kwargs):
        """Creates a decoder block based on the MobileNetV2 bottleneck block."""
        super(DecoderBlock, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.pad1 = ReflectionPad2D([[0, 0], [0, 0]],
                                    name=name + '_expand_reflection_pad')
        self.conv1 = layers.Conv2D(self.filters[0],
                                   kernel_size=(1, 1),
                                   padding='valid',
                                   data_format='channels_last',
                                   activation='linear',
                                   name=name + '_expand')
        self.relu1 = layers.ReLU(max_value=6., name=name + '_expand_relu')
        self.pad2 = ReflectionPad2D([[1, 1], [1, 1]],
                                    name=name + '_depthwise_reflection_pad')
        self.conv2 = layers.DepthwiseConv2D(kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding='valid',
                                            activation='linear',
                                            data_format='channels_last',
                                            name=name + '_depthwise')
        self.relu2 = layers.ReLU(max_value=6., name=name + '_depthwise_relu')
        self.pad3 = ReflectionPad2D([[0, 0], [0, 0]],
                                    name=name + '_project_reflection_pad')
        self.conv3 = layers.Conv2D(self.filters[1],
                                   kernel_size=(1, 1),
                                   padding='valid',
                                   data_format='channels_last',
                                   activation='linear',
                                   name=name + '_project')

    def call(self, x):
        """Do a forward pass of the Decoder block on a tensor."""
        output = self.pad1(x)
        output = self.conv1(output)
        output = self.relu1(output)
        output = self.pad2(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pad3(output)
        output = self.conv3(output)
        return output

    def get_config(self):
        """Get a dict of assigned options for this layer."""
        base_config = super(DecoderBlock, self).get_config()
        config = {'filters': self.filters, **base_config}
        return config


class TVLoss(layers.Layer):
    """A layer that computes TVLoss as a weighted sum of content and style loss."""
    def __init__(self,
                 content_weight=1,
                 style_weight=1e-2,
                 name='tv_loss',
                 **kwargs):
        """Create a TVLoss layer."""
        super(TVLoss, self).__init__(self, name=name, **kwargs)
        self.content_weight = content_weight
        self.style_weight = style_weight

    def call(self, x):
        """Compute the total TVLoss."""
        style_features, output_features, target = x
        return self.content_weight * content_loss(
            target, output_features[-1]) + self.style_weight * style_loss(
                style_features, output_features)
