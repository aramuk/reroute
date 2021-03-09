##########################################################
# A style transfer model inspired by AdaIN-style         #
# Using a MobileNetV2 encoder and trained on Cityscapes. #
##########################################################

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.layers as layers


class AdaIN(layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super(AdaIN, self).__init__(name=self.__class__.__name__, **kwargs)
        self.epsilon = epsilon

    def call(self, x):
        content, style = x
        mean_C, var_C = tf.nn.moments(content, axis=[1, 2], keepdims=True)
        mean_S, var_S = tf.nn.moments(style, axis=[1, 2], keepdims=True)
        std_C = tf.math.sqrt(var_C + self.epsilon)
        std_S = tf.math.sqrt(var_S + self.epsilon)
        return std_S * (content - mean_C) / std_C + mean_S


class MobileAdaIN(tf.keras.Model):

    INPUT_SHAPE = (513, 513, 3)

    def __init__(self):
        self.backbone = MobileNetV2(input_shape=self.INPUT_SHAPE,
                                    weights="imagenet")
        self.upsample = layers.UpSampling2D(size=(2, 2))
        self.adain = AdaIN()
        for layer in self.backbone.layers:
            print(layer)

    def call(self, x):
        return x
