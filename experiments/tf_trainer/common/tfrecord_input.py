"""DatasetInput class based on TFRecord files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import tensorflow as tf
from typing import Callable, List

from tf_trainer.common import base_model
from tf_trainer.common import dataset_input
from tf_trainer.common import types


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


class TFRecordInput(dataset_input.DatasetInput):
  """Simple no-preprocessing TFRecord based DatasetInput.

  Handles parsing of TF Examples.

  Regardless of which TF Example feature key is used, as specified by the
  FLAGS.text_feature, the simple input will store the input text feature in
  the feature key _text_feature.
  """

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

  def _process_labels(self, features, parsed):
    labels = {}
    for label in self._labels:
      label_value = parsed[label]
      # Missing weights are negative, find them and zero those features out.
      weight = tf.cast(tf.greater_equal(label_value, 0.0), dtype=tf.float32)
      if self._round_labels:
        label_value = tf.round(label_value)
      features[label + '_weight'] = weight
      labels[label] = tf.multiply(label_value, weight)
    return features, labels

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

    features = {base_model.TEXT_FEATURE_KEY: parsed[self._text_feature]}
    return self._process_labels(features, parsed)


class TFRecordInputWithTokenizer(TFRecordInput):
  """TFRecord based DatasetInput.

  Handles parsing of TF Examples.

  When handling text input, this class will rewrite the text input future,
  using the preprocessing fn. That is, the text feature will be rewritten
  as a new key in the output changing both the type and contents - from
  a string to a tensor of in integers representing tokens of some kind.
  TODO: preserve the original string and write a new key.
  """

  def __init__(self,
               train_preprocess_fn: Callable[[str], List[str]],
               batch_size: int = 64,
               num_prefetch: int = 5,
               max_seq_len: int = 30000) -> None:
    super().__init__(batch_size, num_prefetch)
    self._train_preprocess_fn = train_preprocess_fn
    self._max_seq_len = max_seq_len

  def _input_fn_from_file(self, filepath: str) -> types.FeatureAndLabelTensors:

    filenames_dataset = tf.data.Dataset.list_files(filepath)
    filenames_dataset = filenames_dataset.repeat(None)
    dataset = tf.data.TFRecordDataset(filenames_dataset) # type: tf.data.TFRecordDataset

    parsed_dataset = dataset.map(
        self._read_tf_example, num_parallel_calls=multiprocessing.cpu_count())
    parsed_dataset = parsed_dataset.filter(
        lambda x, _: tf.less(x['sequence_length'], self._max_seq_len))

    padded_shapes = ({
        base_model.TOKENS_FEATURE_KEY: [None],
        'sequence_length': []
    }, {label: [] for label in self._labels})
    parsed_dataset = parsed_dataset.apply(
        tf.contrib.data.bucket_by_sequence_length(
            element_length_func=lambda x, _: x['sequence_length'],
            bucket_boundaries=[(i + 1) * 20 for i in range(10)],
            bucket_batch_sizes=[self._batch_size] * 11,
            padded_shapes=padded_shapes))
    batched_dataset = parsed_dataset.prefetch(self._num_prefetch)
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
    keys_to_features[self.text_feature()] = tf.FixedLenFeature([], tf.string)
    for label in self._labels:
      keys_to_features[label] = tf.FixedLenFeature([], tf.float32, -1.0)
    parsed = tf.parse_single_example(
        record, keys_to_features)  # type: Dict[str, types.Tensor]

    text = parsed[self.text_feature()]
    tokens = self._train_preprocess_fn(text)
    features = {
        base_model.TOKENS_FEATURE_KEY: tokens,
        'sequence_length': tf.shape(tokens)[0],
    }
    return self._process_labels(features, parsed)
