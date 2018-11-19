"""DatasetInput class based on TFRecord files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import tensorflow as tf
from tf_trainer.common import dataset_input
from tf_trainer.common import types
from typing import Dict
from typing import List


tf.app.flags.DEFINE_string('train_path', None,
                           'Path to the training data TFRecord file.')
tf.app.flags.DEFINE_string('validate_path', None,
                           'Path to the validation data TFRecord file.')
tf.app.flags.DEFINE_string('labels', 'frac_neg',
                           'Comma separated list of (float) label features.')
tf.app.flags.DEFINE_string('text_feature', 'comment_text',
                           'Name of feature containing text input.')
tf.app.flags.DEFINE_boolean('round_labels', True,
                            'Round label features to 0 or 1 if true.')


FLAGS = tf.app.flags.FLAGS


class TFSimpleRecordInput(dataset_input.DatasetInput):
  """Simple no-preprecoessing TFRecord based DatasetInput.

  Handles parsing of TF Examples.

  Regardless of which TF Example feature key is used, as specified by the
  FLAGS.text_feature, the simple input will store the input text feature in
  the feature key _text_feature.
  """
  _output_text_feature = 'text'

  def __init__(self,
               batch_size: int = 64,
               num_prefetch: int = 5) ->None:
    self._labels = FLAGS.labels.split(',')
    self._batch_size = batch_size
    self._num_prefetch = num_prefetch
    self._text_feature = FLAGS.text_feature
    self._round_labels = FLAGS.round_labels

  def labels(self) -> List[str]:
    """List of the names of the float label features."""
    return self._labels

  def text_feature(self) -> str:
    """Name of the feature containing the input text from examples."""
    return self._text_feature

  def train_input_fn(self) -> types.FeatureAndLabelTensors:
    """input_fn for TF Estimators for training set."""
    assert FLAGS.train_path
    return self._input_fn_from_file(FLAGS.train_path)

  def validate_input_fn(self) -> types.FeatureAndLabelTensors:
    """input_fn for TF Estimators for validation set."""
    assert FLAGS.validate_path
    return self._input_fn_from_file(FLAGS.validate_path)

  def _input_fn_from_file(self, filepath: str) -> types.FeatureAndLabelTensors:
    dataset = tf.data.TFRecordDataset(filepath)  # type: tf.data.TFRecordDataset
    parsed_dataset = dataset.map(
        self._read_tf_example,
        num_parallel_calls=multiprocessing.cpu_count())
    batched_dataset = parsed_dataset.batch(self._batch_size)
    batched_dataset = batched_dataset.prefetch(self._num_prefetch)
    return batched_dataset

  def _read_tf_example(
      self,
      record: tf.Tensor,
  ) -> types.FeatureAndLabelTensors:
    """Parses TF Example protobuf into a text feature and labels.

    The input TF Example has a text feature as a singleton list with the full
    comment as the single element.
    """

    keys_to_features = {}
    keys_to_features[self._text_feature] = tf.FixedLenFeature([], tf.string)
    for label in self._labels:
      keys_to_features[label] = tf.FixedLenFeature([], tf.float32, -1.0)
    parsed = tf.parse_single_example(
        record, keys_to_features)  # type: Dict[str, types.Tensor]

    features = {self._output_text_feature: parsed[self._text_feature]}
    labels = {}
    for label in self._labels:
      labels[label] = parsed[label]
      if self._round_labels:
        labels[label] = tf.round(labels[label])
    return features, labels
