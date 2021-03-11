##########################################################
# A style transfer model inspired by AdaIN-style         #
# Using a MobileNetV2 encoder and trained on Cityscapes. #
##########################################################

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2

from encoder import Encoder
from decoder import Decoder
from layers import AdaIN, TVLoss
from losses import content_loss, style_loss


def build_model(input_shape=(None, 224, 224, 3), **kwargs):
    # Build layers of model
    encoder = Encoder(**kwargs.get('encoder_kwargs', {}))
    decoder = Decoder(**kwargs.get('decoder_kwargs', {}))
    adain = AdaIN()
    # tv_loss = TVLoss()
    # Define inputs
    content_input = layers.Input(shape=input_shape, name='content_image')
    style_input = layers.Input(shape=input_shape, name='style_image')
    # Define forward propagation
    content_features = encoder(content_input)
    style_features = encoder(style_input)
    target = adain([content_features[-1], style_features[-1]])
    output = decoder(target)
    # Compute loss
    output_encodings = encoder(output)
    # loss = tv_loss([style_features, output_encodings, target])
    # Build model
    model = models.Model(inputs=[content_input, style_input],
                         outputs=[
                             output, target, content_features, style_features,
                             output_encodings
                         ])
    return model


if __name__ == '__main__':
    INPUT_SHAPE = (224, 224, 3)
    EXPECTED_TARGET_SHAPE = (None, 14, 14, 96)
    EXPECTED_OUTPUT_SHAPE = (None, *INPUT_SHAPE)

    encoder = Encoder()
    decoder = Decoder()
    adain = AdaIN()

    content_input = layers.Input(shape=INPUT_SHAPE, name='content_image')
    style_input = layers.Input(shape=INPUT_SHAPE, name='style_image')

    def has_shape(target, shape):
        return all(dim == target_dim
                   for dim, target_dim in zip(target.shape, shape))

    content_features = encoder(content_input)
    assert len(
        content_features
    ) == 5, 'Encoder returns {} content feature maps; expected 5'.format(
        len(content_features))
    style_features = encoder(style_input)
    assert len(
        style_features
    ) == 5, 'Encoder returns {} style feature maps; expected 5'.format(
        len(style_features))
    target = adain([content_features[-1], style_features[-1]])
    assert has_shape(
        target,
        EXPECTED_TARGET_SHAPE), 'target has shape {}; expected {}'.format(
            target.shape, EXPECTED_TARGET_SHAPE)
    output = decoder(target)
    assert has_shape(
        output,
        EXPECTED_OUTPUT_SHAPE), 'output has shape {}; expected {}'.format(
            output.shape, EXPECTED_OUTPUT_SHAPE)
    model = models.Model(inputs=[content_input, style_input], outputs=[output])
    for var in model.trainable_weights:
        print(var.name, var.shape)