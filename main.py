#!/usr/bin/env python3.8

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from lib.model import DeepLabV3, download_model
from lib.visualize import vis_segmentation


def save_model(model):
    image = Image.open(
        'cityscapes/leftImg8bit/test/berlin/berlin_000032_000019_leftImg8bit.png'
    )
    resized_im, batch_seg_map = model.run(image)
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()

    save_dir = 'models/mobilenetv2_coco_cityscapes_trainfine'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    tf.compat.v1.saved_model.simple_save(
        model.sess, save_dir,
        {model.INPUT_TENSOR_NAME: tf.convert_to_tensor(resized_im)},
        {model.OUTPUT_TENSOR_NAME: tf.convert_to_tensor(batch_seg_map)})
    print('{} model saved to {}'.format(model.__class__, save_dir))


if __name__ == '__main__':
    tarball_path = download_model()
    model = DeepLabV3(tarball_path)

    dataset, info = tfds.load('cityscapes/semantic_segmentation', split='train', with_info=True)

    example = next(iter(dataset))
    print(example['image_id'])
    print(example['image_left'])
    print(example['segmentation_label'])

    tfds.visualization.show_examples(dataset, info, row=3, col=3)

    # fig, ax = plt.subplots(5, 4)
    # for i, example in enumerate(dataset):
    #     if i == 5:
    #         break
    #     image = example['image_left']
    #     resized_im, batch_seg_map = model.run(image)
    #     target = example['segmentation_label']
    #     # target = np.asarray(Image.fromarray(example['segmentation_label'].numpy).resize(
    #     #     resized_im.shape, Image.ANTIALIAS))
    #     # Display
    #     ax[i][0].imshow(image)
    #     ax[i][1].imshow(batch_seg_map[0])
    #     ax[i][2].imshow(target)
    #     acc = (batch_seg_map[0] == target)
    #     print(type(target), type(batch_seg_map[0]))
    #     print(batch_seg_map[0].shape, target.shape, type(acc), acc.shape)
    #     ax[i][3].imshow(acc)
    # plt.show()
