##########################################################
# A MobileNet-based AdaIN encoder network.               #
##########################################################

from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.layers as layers

from layers import ReflectionPad2D, DecoderBlock


class Decoder(layers.Layer):
    """An decoder network based on MobileNetV2 with reflection padding."""
    def __init__(self, name='mobilenetv2_decoder', **kwargs):
        """Creates the encoder with."""
        super(Decoder, self).__init__(name=name, **kwargs)
        self._layers = [
            DecoderBlock([96, 64], name='decode_1'),
            DecoderBlock([64, 64], name='decode_2'),
            DecoderBlock([64, 64], name='decode_3'),
            DecoderBlock([64, 64], name='decode_4'),
            DecoderBlock([64, 32], name='decode_5'),
            DecoderBlock([32, 32], name='decode_6'),
            DecoderBlock([32, 32], name='decode_7'),
            DecoderBlock([32, 24], name='decode_8'),
            DecoderBlock([24, 24], name='decode_9'),
            DecoderBlock([24, 16], name='decode_10'),
            ReflectionPad2D(name='decode_reflection_pad'),
            layers.Conv2D(3, (3, 3),
                          padding='valid',
                          activation='linear',
                          data_format='channels_last',
                          name='decode_conv')
        ]

    def call(self, x):
        """Do a forward pass of the decoder on a tensor."""
        output = x
        for layer in self._layers:
            output = layer(output)
        return output


if __name__ == '__main__':
    block = Decoder()
    # for layer in block._layers:
    #     print(layer.name, layer.get_config())

    import tensorflow as tf
    input_shape = (1, 14, 14, 96)
    test = tf.random.uniform(input_shape)
    assert test.shape == input_shape
    output = block.call(test)
    print(output.shape)