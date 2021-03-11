##########################################################
# A MobileNet-based AdaIN encoder network.               #
##########################################################

import os

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from layers import ReflectionPad2D


class Encoder(keras.Model):
    """An encoding network based on MobileNetV2 with reflection padding."""
    def __init__(self,
                 name='mobilenetv2_encoder',
                 cutoff_layer='block_10_project_BN',
                 **kwargs):
        """Creates the encoder with."""
        super(Encoder, self).__init__(name=name, **kwargs)
        backbone = keras.applications.MobileNetV2()
        last_layer = backbone.get_layer(cutoff_layer)
        self._layers = []
        for layer in backbone.layers:
            config = layer.get_config()
            if config.get('padding') == 'same':
                padding = [[0, 0], [0, 0]] if config['kernel_size'] == (
                    1, 1) else [[1, 1], [1, 1]]
                self._layers.append(
                    ReflectionPad2D(padding,
                                    name=layer.name + '_reflection_pad'))
                valid_conv = type(layer).from_config({
                    **config, 'padding': 'valid',
                    'trainable': False,
                    'kernel_initializer': keras.initializers.Constant(
                        layer.get_weights())
                })
                assert valid_conv.get_config()['padding'] == 'valid'
                self._layers.append(valid_conv)
            else:
                self._layers.append(layer)
                layer.trainable = False
            if layer == last_layer:
                break

    def call(self, x):
        """Encodes a single tensor."""
        output = x
        residual = None
        for layer in self._layers:
            if layer.name.endswith('_add'):
                output = layer([residual, output])
            else:
                output = layer(output)
            if layer.name.endswith('_project_BN'):
                residual = output
        return output


if __name__ == '__main__':
    block = Encoder()

    import tensorflow as tf
    input_shape = (10, 224, 224, 3)
    X = tf.random.uniform(input_shape)
    assert X.shape == input_shape
    Y = block.call(X)
    assert Y.shape == (10, 14, 14, 96)