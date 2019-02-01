import tensorflow as tf
from pathlib import Path

import os
import random

from tf_trainer.common import dataset_input
from tf_trainer.common import types
from typing import List, Dict, Tuple

tf.app.flags.DEFINE_string('train_path', None,
                           'Path to the training data TFRecord file.')
tf.app.flags.DEFINE_string('dev_path', None,
                           'Path to the training data TFRecord file.')
tf.app.flags.DEFINE_string('episode_size', None,
                           'Path to the training data TFRecord file.')


class EpisodicTFRecordInput(dataset_input.DatasetInput):

  def __init__(self, train_dir, validate_dir) -> None:
    self.train_dir = train_dir
    self.validate_dir = validate_dir

  def train_input_fn(self) -> types.FeatureAndLabelTensors:
    all_episodes = self._get_randomized_episodes(self.train_dir)
    all_texts = [ep[0] for ep in all_episodes]
    all_domains = [ep[1] for ep in all_episodes]
    all_labels = [ep[2] for ep in all_episodes]
    ds = tf.data.Dataset.from_tensor_slices((all_texts, all_domains,
                                             all_labels))
    self.episode_batches_itr = ds.make_one_shot_iterator()
    return self.episode_batches_itr.get_next()

  def validate_input_fn(self) -> types.FeatureAndLabelTensors:
    pass

  def _get_randomized_episodes(self, directory: str):
    """Retrieves a list of domain specific datasets.

    Given a directory of TFRecord files, each holding data for a given domain,
    with file name "[domain].tfrecord", returns an iterator of datasets, each
    corresponding to the data for a single domain.
    """

    tfrecord_files = tf.gfile.Glob(os.path.join(directory, '*.tfrecord'))
    episodes = []
    for file_no, tfrecord_file in enumerate(tfrecord_files):
      print('PROCESSING FILE {}: {}'.format(file_no, tfrecord_file))
      episodes.append(self._dataset_from_tfrecord_file(tfrecord_file))
    random.shuffle(episodes)
    return episodes

  def _dataset_from_tfrecord_file(
      self,
      tfrecord_file: str) -> Tuple[List[tf.Tensor], List[str], List[tf.Tensor]]:
    # The domain happens to be the stem for this dataset.
    domain = Path(tfrecord_file).stem

    def _read_tf_example(record):
      parsed = tf.parse_single_example(
          record, {
              'text': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64)
          })  # type: Dict[str, types.Tensor]

      return {
          'text': parsed['text'],
          'domain': domain,
          'label': parsed['label'],
      }

    examples = list(tf.python_io.tf_record_iterator(tfrecord_file))
    random.shuffle(examples)

    datapoints = [_read_tf_example(example) for example in examples]
    texts = [dp['text'] for dp in datapoints]
    domains = [dp['domain'] for dp in datapoints]
    labels = [dp['label'] for dp in datapoints]
    return (texts, domains, labels)
