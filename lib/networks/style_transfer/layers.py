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


class MNetV2_Block(layers.Layer):
    """A bottleneck block, as articulated in the MobileNetV2 paper."""
    def __init__(self,
                 input_size,
                 output_size,
                 filters,
                 name='mnetv2_block',
                 **kwargs):
        """Creates a MobileNetV2 bottleneck block."""
        super(MNetV2_Block, self).__init__(name=name, **kwargs)
        self.pad1 = ReflectionPad2D(name=name + '_expand_reflection_pad')
        self.conv1 = layers.Conv2D(filters[0],
                                   kernel_size=(1, 1),
                                   padding='valid',
                                   data_format='channels_last',
                                   activation='linear',
                                   name=name + '_expand')
        self.bn1 = layers.BatchNormalization(axis=3,
                                             momentum=0.999,
                                             epsilon=0.001,
                                             name=name + '_expand_BN')
        self.relu1 = layers.ReLU(max_value=6., name=name + '_expand_relu')
        self.pad2 = layers.ZeroPadding2D(
            padding=((0, 1), (0, 1)),
            data_format='channels_last',
            name=name + '_pad',
        )
        self.conv2 = layers.DepthwiseConv2D(kernel_size=(3, 3),
                                            strides=(2, 2),
                                            padding='valid',
                                            activation='linear',
                                            data_format='channels_last',
                                            name=name + '_depthwise')
        self.bn2 = layers.BatchNormalization(axis=3,
                                             momentum=0.999,
                                             epsilon=0.001,
                                             name=name + '_depthwise_BN')
        self.relu2 = layers.ReLU(max_value=6., name=name + '_depthwise_relu')
        self.pad3 = ReflectionPad2D(name=name + '_project_reflection_pad')
        self.conv3 = layers.Conv2D(filters[1],
                                   kernel_size=(1, 1),
                                   padding='valid',
                                   data_format='channels_last',
                                   activation='linear',
                                   name=name + '_project')
        self.bn3 = layers.BatchNormalization(axis=3,
                                             momentum=0.999,
                                             epsilon=0.001,
                                             name=name + '_project_BN')

    def call(self, x):
        """Do a forward pass of the MobileNetV2 block on a tensor."""
        output = self.pad1(x)
        output = self.conv1(output)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pad2(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu3(output)
        output = self.pad3(output)
        output = self.conv3(output)
        output = self.bn3(output)
        return output