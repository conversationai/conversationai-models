"""DatasetInput class based on TFRecord files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import tensorflow as tf
from tf_trainer.common import dataset_input
from tf_trainer.common import types
from typing import Callable, Dict, List


class TFRecordInput(dataset_input.DatasetInput):
  """TFRecord based DatasetInput.

  Handles parsing of TF Examples.
  """

  def __init__(self,
               train_path: str,
               validate_path: str,
               text_feature: str,
               labels: Dict[str, tf.DType],
               train_preprocess_fn: Callable[[str], List[str]],
               batch_size: int = 64,
               round_labels: bool = True,
               num_prefetch: int = 5,
               max_seq_len: int = 30000) -> None:
    self._train_path = train_path
    self._validate_path = validate_path
    self._text_feature = text_feature
    self._labels = labels
    self._batch_size = batch_size
    self._train_preprocess_fn = train_preprocess_fn
    self._round_labels = round_labels
    self._num_prefetch = num_prefetch
    self._max_seq_len = max_seq_len

  def train_input_fn(self) -> types.FeatureAndLabelTensors:
    """input_fn for TF Estimators for training set."""
    return self._input_fn_from_file(self._train_path)

  def validate_input_fn(self) -> types.FeatureAndLabelTensors:
    """input_fn for TF Estimators for validation set."""
    return self._input_fn_from_file(self._validate_path)

  def _input_fn_from_file(self, filepath: str) -> types.FeatureAndLabelTensors:

    filenames_dataset = tf.data.Dataset.list_files(filepath)
    filenames_dataset = filenames_dataset.repeat(None)
    dataset = tf.data.TFRecordDataset(filenames_dataset) # type: tf.data.TFRecordDataset

    parsed_dataset = dataset.map(
        self._read_tf_example, num_parallel_calls=multiprocessing.cpu_count())

    padded_shapes = ({
        self._text_feature: [None],
        'sequence_length': []
    }, {label: [] for label in self._labels})
    parsed_dataset = parsed_dataset.filter(
        lambda x, _: x['sequence_length'] < self._max_seq_len)
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
    DEFAULT_VALUES = {tf.string: '', tf.float32: -1.0, tf.int32: -1}

    keys_to_features = {}
    keys_to_features[self._text_feature] = tf.FixedLenFeature([], tf.string)
    for label, dtype in self._labels.items():
      keys_to_features[label] = tf.FixedLenFeature([], dtype,
                                                   DEFAULT_VALUES[dtype])
    parsed = tf.parse_single_example(
        record, keys_to_features)  # type: Dict[str, types.Tensor]

    text = parsed[self._text_feature]
    features = {self._text_feature: self._train_preprocess_fn(text)}
    features['sequence_length'] = tf.shape(features[self._text_feature])[0]
    if self._round_labels:
      labels = {label: tf.round(parsed[label]) for label in self._labels}
    else:
      labels = {label: parsed[label] for label in self._labels}

    return features, labels
