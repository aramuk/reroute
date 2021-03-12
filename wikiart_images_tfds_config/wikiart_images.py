"""wikiart_images dataset."""

import tensorflow_datasets.public_api as tfds

# TODO(wikiart_images): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
A subset of the ArtGAN dataset, obtained from https://www.kaggle.com/c/painter-by-numbers/.
"""

# TODO(wikiart_images): BibTeX citation
_CITATION = """
@article{artgan2018,
  title={Improved ArtGAN for Conditional Synthesis of Natural Image and Artwork},
  author={Tan, Wei Ren and Chan, Chee Seng and Aguirre, Hernan and Tanaka, Kiyoshi},
  journal={IEEE Transactions on Image Processing},
  volume    = {28},
  number    = {1},
  pages     = {394--409},
  year      = {2019},
  url       = {https://doi.org/10.1109/TIP.2018.2866698},
  doi       = {10.1109/TIP.2018.2866698}
}
"""


class WikiartImages(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for wikiart_images dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Register into https://www.kaggle.com/ to get the data. Go to the competition
  , and place the `train_1.zip`
  file in the `manual_dir/`.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # yapf: disable
    return tfds.core.DatasetInfo(
        builder=self,
        description="""
        A small subset of the WikiArt dataset, consisting of the first partition retrieved
        from the kaggle competition, "By the Numbers".
        """,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(None, None, 3), encoding_format='jpeg')
        }),
        citation=_CITATION)
    # yapf: enable

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    archive_path = dl_manager.manual_dir / 'train_1.zip'
    extracted_path = dl_manager.extract(archive_path)
    return {'train': self._generate_examples(extracted_path / 'train_1')}

  def _generate_examples(self, path):
    """Yields examples."""
    for image_path in path.glob('*.jpg'):
      yield image_path.name, {
        'image': image_path,
      }
