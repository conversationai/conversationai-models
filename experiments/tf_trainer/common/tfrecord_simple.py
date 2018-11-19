"""DatasetInput class based on TFRecord files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import tensorflow as tf
from tf_trainer.common import dataset_input
from tf_trainer.common import types
from typing import Dict


class TFSimpleRecordInput(dataset_input.DatasetInput):
  """Simple no-preprecoessing TFRecord based DatasetInput.

  Handles parsing of TF Examples.
  """

  def __init__(self,
               train_path: str,
               validate_path: str,
               text_feature: str,
               labels: Dict[str, tf.DType],
               batch_size: int = 64,
               round_labels: bool = True,
               num_prefetch: int = 5) ->None:
    self._train_path = train_path
    self._validate_path = validate_path
    self._text_feature = text_feature
    self._labels = labels
    self._batch_size = batch_size
    self._round_labels = round_labels
    self._num_prefetch = num_prefetch

  def train_input_fn(self) -> types.FeatureAndLabelTensors:
    """input_fn for TF Estimators for training set."""
    return self._input_fn_from_file(self._train_path)

  def validate_input_fn(self) -> types.FeatureAndLabelTensors:
    """input_fn for TF Estimators for validation set."""
    return self._input_fn_from_file(self._validate_path)

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
    DEFAULT_VALUES = {tf.string: '', tf.float32: -1.0, tf.int32: -1}

    keys_to_features = {}
    keys_to_features[self._text_feature] = tf.FixedLenFeature([], tf.string)
    for label, dtype in self._labels.items():
      keys_to_features[label] = tf.FixedLenFeature([], dtype,
                                                   DEFAULT_VALUES[dtype])
    parsed = tf.parse_single_example(
        record, keys_to_features)  # type: Dict[str, types.Tensor]

    features = {self._text_feature: parsed[self._text_feature]}
    labels = {}
    for label in self._labels:
      labels[label] = parsed[label]
      if self._round_labels:
        labels[label] = tf.round(labels[label])
    return features, labels
