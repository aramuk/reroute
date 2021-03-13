##########################################################
# A MobileNet-based AdaIN encoder network.               #
##########################################################

import os

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from .layers import ReflectionPad2D


class Encoder(layers.Layer):
    """An encoding network based on MobileNetV2 with reflection padding."""
    def __init__(self,
                 name='mobilenetv2_encoder',
                 output_layers=[
                     'expanded_conv_project_BN', 'block_1_project_BN',
                     'block_3_project_BN', 'block_6_project_BN',
                     'block_10_project_BN'
                 ],
                 cutoff_layer='block_10_project_BN',
                 **kwargs):
        """Creates the encoder with."""
        super(Encoder, self).__init__(name=name, **kwargs)
        self.output_layers = set(output_layers)
        self.cutoff_layer = cutoff_layer
        # Transfer learn a model from MobileNetV2()
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
        z = x
        residual = None
        outputs = []
        for layer in self._layers:
            if layer.name.endswith('_add'):
                z = layer([residual, z])
            else:
                z = layer(z)
            if layer.name.endswith('_project_BN'):
                residual = z
            if layer.name in self.output_layers:
                outputs.append(z)
        return outputs
    
    def get_config(self):
        """Get a dict of assigned options for this layer."""
        base_config = super(Encoder, self).get_config()
        config = {
            'output_layers': list(self.output_layers), 
            'cutoff_layer': self.cutoff_layer, 
            **base_config
        }
        return config


if __name__ == '__main__':
    block = Encoder()

    import tensorflow as tf
    input_shape = (10, 224, 224, 3)
    X = tf.random.uniform(input_shape)
    assert X.shape == input_shape
    Y = block(X)
    assert Y.shape == (10, 14, 14, 96)