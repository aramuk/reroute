##########################################################
# A MobileNet-based AdaIN encoder network.               #
##########################################################

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.layers as layers

from layers import ReflectionPad2D


class Encoder(layers.Layer):
    """An encoding network based on MobileNetV2 with reflection padding."""
    def __init__(self,
                 input_shape=(None, 513, 256, 3),
                 name='mobilenetv2_encoder',
                 cutoff_layer='block_10_project_BN',
                 **kwargs):
        """Creates the encoder with."""
        super(Encoder, self).__init__(name=name, **kwargs)
        backbone = MobileNetV2(weights='imagenet')
        last_layer = backbone.get_layer(cutoff_layer)
        self._layers = []
        for layer in backbone.layers:
            layer.trainable = False
            if layer.get_config().get('padding', None) == 'same':
                self._layers.append(
                    ReflectionPad2D(name=layer.name + '_reflection_pad'))
                layer.get_config()['padding'] = 'valid'
            self._layers.append(layer)
            if layer == last_layer:
                break

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
