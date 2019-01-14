"""TODO(jjtan): DO NOT SUBMIT without one-line documentation for episodic_tfrecord_input.

TODO(jjtan): DO NOT SUBMIT without a detailed description of
episodic_tfrecord_input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pathlib import Path

import functools
import os
import random

from tf_trainer.common import base_model
from tf_trainer.common import dataset_input
from tf_trainer.common import types
from typing import List, Dict

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

  def initialize(self, session) -> None:
    session.run(self.episode_batches_itr.initializer, {})

  def train_input_fn(self) -> types.FeatureAndLabelTensors:
    domains = self._get_domains(self.train_dir)
    num_domains = len(domains)
    texts = tf.stack([domain['text'] for domain in domains])
    labels = tf.stack([domain['label'] for domain in domains])
    randomized_idxs = tf.data.Dataset.range(num_domains).shuffle(
        buffer_size=128)
    episodes = randomized_idxs.map(lambda idx: (texts[idx], labels[idx]))
    episode_batches = episodes.batch(4)
    self.episode_batches_itr = episode_batches.make_initializable_iterator()
    return self.episode_batches_itr.get_next()

  def validate_input_fn(self) -> types.FeatureAndLabelTensors:
    pass

  def _get_domains(self, directory: str) -> List[tf.data.Dataset]:
    """Retrieves an iterator of domain specific datasets.

    Given a directory of TFRecord files, each holding data for a given domain,
    with file name "[domain].tfrecord", returns an iterator of datasets, each
    corresponding to the data for a single domain.
    """

    tfrecord_files = tf.gfile.Glob(os.path.join(directory, '*.tfrecord'))
    domain_datasets = []
    for tfrecord_file in tfrecord_files:
      domain_datasets.append(self._dataset_from_tfrecord_file(tfrecord_file))

    # Make episodes of size 32.
    return [
        ds.batch(32).make_one_shot_iterator().get_next()
        for ds in domain_datasets
    ]

  def _dataset_from_tfrecord_file(self, tfrecord_file: str) -> tf.data.Dataset:
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

    return tf.data.TFRecordDataset(tfrecord_file).shuffle(
        buffer_size=128).map(_read_tf_example).repeat()
