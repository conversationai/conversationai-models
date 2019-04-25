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

import bert
import collections
import os
import random

from bert import run_classifier
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


class BertEpisodicInput(object):
  """Generates episodic data."""

  def __init__(self, train_dir, validate_dir, tokenizer) -> None:
    self.train_dir = train_dir
    self.validate_dir = validate_dir
    self._tokenizer = tokenizer

  def train_input_fn(self, tokenizer):
    all_episodes = self._get_randomized_episodes(self.train_dir)
    return all_episodes

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
    for file_no, tfrecord_file in enumerate(tfrecord_files[:100]):
      tf.logging.info('PROCESSING FILE {}: {}'.format(file_no, tfrecord_file))
      episodes.append(self._dataset_from_tfrecord_file(tfrecord_file))

    tf.logging.info('Shuffling episodes')
    random.shuffle(episodes)  # In place shuffle.

    return episodes

  def _dataset_from_tfrecord_file(self, tfrecord_file: str
                                 ) -> List[bert.run_classifier.InputExample]:
    # The domain happens to be the file stem.
    domain = Path(tfrecord_file).stem

    def _create_features(idx, text, label):
      features = bert.run_classifier.convert_single_example(
          idx,
          bert.run_classifier.InputExample(
              guid=None, text_a=text, text_b=None, label=label), [0, 1], 256,
          self._tokenizer)
      return [
          features.input_ids, features.input_mask, features.segment_ids,
          features.label_id, features.is_real_example
      ]

    def _read_tf_example(idx, record):
      parsed = tf.parse_single_example(
          record, {
              'text': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64)
          })  # type: Dict[str, types.Tensor]
      features = tf.py_func(_create_features,
                            [idx, parsed['text'], parsed['label']],
                            [tf.int64, tf.int64, tf.int64, tf.int64, tf.bool])

      return {
          'input_ids': features[0],
          'input_mask': features[1],
          'segment_ids': features[2],
          'label_id': features[3],
          'is_real_example': features[4]
      }

    examples = tf.data.TFRecordDataset(tfrecord_file).apply(
        tf.contrib.data.enumerate_dataset(0)).map(_read_tf_example).shuffle(
            buffer_size=32)

    return examples.make_one_shot_iterator().get_next()
