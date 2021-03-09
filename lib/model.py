#
#
#

import os
import tarfile
import tempfile
import urllib

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf


class DeepLabV3(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.compat.v1.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(
                    file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.compat.v1.graph_util.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
        image: A PIL.Image object, raw input image.

        Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_AREA)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        return resized_image, batch_seg_map


def download_model():
    MODEL_NAME = 'mobilenetv2_coco_cityscapes_trainfine'
    MODEL_URL = 'deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz'
    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _TARBALL_NAME = 'deeplab_model.tar.gz'
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    # Find download directory
    model_dir = './models'
    tf.compat.v1.gfile.MakeDirs(model_dir)
    download_path = os.path.join(model_dir, _TARBALL_NAME)

    if not os.path.exists(download_path):
        # Download saved model
        print('downloading model, this might take a while...')
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + MODEL_URL,
                                   download_path)
        print('download completed! loading DeepLab model...')
    return download_path