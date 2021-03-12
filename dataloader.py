##########################################################
# Datasets and Dataloaders for Cityscapes and Wikiart.   #
##########################################################

import tensorflow as tf
import tensorflow_datasets as tfds

if __name__ == '__main__':
    wikiart, wikiart_info = tfds.load('wikiart_images',
                                      split='train[:2975]',
                                      with_info=True)
    print('WikiArt has {} examples of: {}'.format(len(wikiart),
                                                  str(wikiart.element_spec)))
    wikiart = wikiart.map(lambda d: tf.image.resize_with_pad(
        d['image'], target_height=256, target_width=256)).batch(8)

    print('WikiArt has {} batches of size 8'.format(len(wikiart)))
    for example in wikiart:
        assert example.shape[1] == 256 and example.shape[
            2] == 256, 'WikiArt example has unexpected shape: {}'.format(
                example.shape)
    print('WikiArt was loaded without errors')

    cityscapes, cityscapes_info = tfds.load('cityscapes/semantic_segmentation',
                                            split='train',
                                            with_info=True)
    print('Cityscapes has {} examples of: {}'.format(
        len(cityscapes), str(cityscapes.element_spec)))
    cityscapes = cityscapes.map(lambda d: tf.image.resize_with_pad(
        d['image_left'], target_height=256, target_width=512)).batch(8)

    print('Cityscapes has {} batches of size 8'.format(len(wikiart)))
    for example in cityscapes:
        assert example.shape[1] == 256 and example.shape[
            2] == 512, 'Cityscapes example has unexpected shape: {}'.format(
                example.shape)
    print('Cityscapes was loaded without errors')