import cv2
import tensorflow as tf
import tensorflow_datasets as tfds

from lib.dataloader import StyleTransferDataLoader
from lib.networks.style_transfer.layers import TVLoss
from lib.networks.style_transfer.network import build_model

if __name__ == '__main__':
    train_loader = StyleTransferDataLoader(
        batch_size=1,
        length=1,
        content_transform=lambda I: tf.image.random_crop(
            tf.image.resize_with_pad(I, target_height=256, target_width=512),
            (224, 224, 3)),
        style_transform=lambda I: tf.image.random_crop(
            tf.image.resize_with_crop_or_pad(I, target_height=256, target_width=256),
            (224, 224, 3)))

    example = next(iter(train_loader))
    print(example[0].shape, example[1].shape)
    print(tf.math.reduce_max(example[0]), tf.math.reduce_min(example[0]))
    cv2.imwrite('input_c.png', example[0].numpy()[0] * 255)
    cv2.imwrite('input_s.png', example[1].numpy()[0] * 255)
    MobileAdaIN = build_model(input_shape=(224, 224, 3))
    output, target, content_features, style_features = MobileAdaIN(example)
    print('Output', output.shape, tf.math.reduce_max(output),
          tf.math.reduce_min(output))
    print('Output', target.shape, tf.math.reduce_max(target),
          tf.math.reduce_min(target))
    cv2.imwrite('output.png', output.numpy()[0] * 255)
