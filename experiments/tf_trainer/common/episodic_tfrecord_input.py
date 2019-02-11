# coding=utf-8
# Copyright 2018 The Conversation-AI.github.io Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DatasetInput implementation for episodic data."""

import tensorflow as tf
from pathlib import Path

import collections
import os
import random

from tf_trainer.common import dataset_input
from tf_trainer.common import types
from typing import List, Dict, Tuple, Union

tf.app.flags.DEFINE_string('train_path', None,
                           'Path to the training data TFRecord file.')
tf.app.flags.DEFINE_string('dev_path', None,
                           'Path to the training data TFRecord file.')
tf.app.flags.DEFINE_string('episode_size', None,
                           'Path to the training data TFRecord file.')

Text = Union[tf.Tensor, str]
Label = Union[tf.Tensor, float]

TextDomainLabel = collections.namedtuple('TextDomainLabel',
                                         ['text', 'domain', 'label'])
EpisodeData = collections.namedtuple('EpisodeData',
                                     ['texts', 'domains', 'labels'])


class EpisodicTFRecordInput(dataset_input.DatasetInput):
  """Generates episodic data."""

  def __init__(self, train_dir, validate_dir) -> None:
    self.train_dir = train_dir
    self.validate_dir = validate_dir

  def train_input_fn(self) -> types.FeatureAndLabelTensors:
    all_episodes = self._get_randomized_episodes(self.train_dir)
    all_texts = [ep.texts for ep in all_episodes]
    all_domains = [ep.domains for ep in all_episodes]
    all_labels = [ep.labels for ep in all_episodes]
    ds = tf.data.Dataset.from_tensor_slices((all_texts, all_domains,
                                             all_labels))
    self.episode_batches_itr = ds.make_one_shot_iterator()
    return self.episode_batches_itr.get_next()

  def validate_input_fn(self) -> types.FeatureAndLabelTensors:
    pass

  def _get_randomized_episodes(self, directory: str) -> List[EpisodeData]:
    """Retrieves a list of domain specific datasets.

    Given a directory of TFRecord files, each holding data for a given domain,
    with file name "[domain].tfrecord", returns an iterator of datasets, each
    corresponding to the data for a single domain.
    """

    tfrecord_files = tf.gfile.Glob(os.path.join(directory, '*.tfrecord'))
    episodes = []
    for file_no, tfrecord_file in enumerate(tfrecord_files):
      tf.logging.info('PROCESSING FILE {}: {}'.format(file_no, tfrecord_file))
      episodes.append(self._dataset_from_tfrecord_file(tfrecord_file))

    tf.logging.info('Shuffling episodes')
    random.shuffle(episodes)  # In place shuffle.

    return episodes

  def _dataset_from_tfrecord_file(self, tfrecord_file: str) -> EpisodeData:
    # The domain happens to be the file stem.
    domain = Path(tfrecord_file).stem

    def _read_tf_example(record) -> TextDomainLabel:
      parsed = tf.parse_single_example(
          record, {
              'text': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64)
          })  # type: Dict[str, types.Tensor]

      return TextDomainLabel(
          text=parsed['text'], domain=domain, label=parsed['label'])

    examples = list(tf.python_io.tf_record_iterator(tfrecord_file))
    random.shuffle(examples)

    datapoints = [_read_tf_example(example) for example in examples]
    return EpisodeData(
        texts=[dp.text for dp in datapoints],
        domains=[dp.domain for dp in datapoints],
        labels=[dp.label for dp in datapoints])
