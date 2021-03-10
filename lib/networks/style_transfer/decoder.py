##########################################################
# A MobileNet-based AdaIN encoder network.               #
##########################################################

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.layers as layers

from layers import ReflectionPad2D


class Decoder(layers.Layer):
    """An decoder network based on MobileNetV2 with reflection padding."""
    def __init__(self,
                 input_shape=(None, 14, 14, 576),
                 name='mobilenetv2_decoder',
                 **kwargs):
        """Creates the encoder with."""
        super(Decoder, self).__init__(name=name, **kwargs)
        self.pad1 = ReflectionPad2D()

    def call(self, x):
        """Runs"""
        output = x
        for layer in self._layers:
            output = layer(output)
        return output


if __name__ == '__main__':
    block = Encoder()
    for layer in block._layers:
        print(layer.name, layer)
