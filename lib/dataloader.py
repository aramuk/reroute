##########################################################
# Datasets and Dataloaders for Cityscapes and Wikiart.   #
##########################################################

import tensorflow as tf
import tensorflow_datasets as tfds


class StyleTransferDataLoader():
    """A dataloader that loads the Cityscapes and WikiArt datasets simultaneously"""

    def __init__(self,
                 batch_size=8,
                 length=2975,
                 content_transform=None,
                 style_transform=None,
                 content_options={},
                 style_options={}):
        """Create and preprocessthe content and style datasets."""
        # Save metrics
        self.batch_size = batch_size
        self.length = length
        # Load and preprocess the content dataset
        self.content_ds = tfds.load(
            'cityscapes/semantic_segmentation',
            split='train[:{}]'.format(length),
            **content_options).map(lambda d: d['image_left']/255)
        if content_transform and callable(content_transform):
            self.content_ds = self.content_ds.map(content_transform)
        self.content_ds = self.content_ds.batch(batch_size)
        # Load and preprocess the style dataset
        self.style_ds = tfds.load('wikiart_images',
                                  split='train[:{}]'.format(length),
                                  **style_options).map(lambda d: d['image']/255)
        if style_transform and callable(style_transform):
            self.style_ds = self.style_ds.map(style_transform)
        self.style_ds = self.style_ds.batch(self.batch_size)

    def __iter__(self):
        """Returns an iterator to the combined dataset."""
        return zip(self.content_ds, self.style_ds)

    def __len__(self):
        """Get the number of batches in the data loader."""
        return min(len(self.content_ds), len(self.style_ds))


if __name__ == '__main__':
    import time
    dataloader = StyleTransferDataLoader(
        content_transform=lambda I: tf.image.resize_with_pad(
            I, target_height=256, target_width=512),
        style_transform=lambda I: tf.image.resize_with_pad(
            I, target_height=256, target_width=256))
    assert len(
        dataloader
    ) == 372, 'Expected dataloader to have 372 batches, received {}'.format(
        len(dataloader))

    print('Loading examples from style transfer dataset......')
    start = time.time()
    for C, S in dataloader:
        assert C.shape[1] == 256 and C.shape[2] == 512, \
            'Cityscapes example has unexpected shape: {}'.format(C.shape)
        assert S.shape[1] == 256 and S.shape[2] == 256, \
            'Cityscapes example has unexpected shape: {}'.format(S.shape)

    print('SUCCESS: Style transfer dataset loaded without errors in {} seconds'.format(
        time.time() - start))
