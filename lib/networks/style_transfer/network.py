##########################################################
# A style transfer model inspired by AdaIN-style         #
# Using a MobileNetV2 encoder and trained on Cityscapes. #
##########################################################

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.layers as layers

from encoder import Encoder
from decoder import Decoder
from layers import AdaIN


class Encoder(tf.keras.Model):
    def __init__(self, **kwargs):
        self.backbone = MobileNetV2(weights="imagenet", **kwargs)
        self.layers = []
        for layer in self.backbone:
            if 'block_11' in layer.name:
                break
            layer.trainable = False
            layers.push(layer)

    def call(self, image):
        output = image
        for layer in self.layers:
            output = layer(output)
        return output


class MobileAdaIN(tf.keras.Model):

    INPUT_SIZE = 224

    def __init__(self, encoder_kwargs):
        self.encoder = Encoder(**encoder_kwargs)
        self.upsample = layers.UpSampling2D(size=(2, 2))
        self.adain = AdaIN()
        for layer in self.backbone.layers:
            print(layer)

    def call(self, image):
        content, style = image
        content_encoded = self.encoder(content)
        style_encoded = self.encoder(style)
        return x
