"""DatasetInput class based on TFRecord files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import tensorflow as tf
from tf_trainer.common import base_model
from tf_trainer.common import dataset_input
from tf_trainer.common import tfrecord_simple
from tf_trainer.common import types

from typing import Callable, Dict, List


class TFRecordInput(tfrecord_simple.TFSimpleRecordInput):
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
    if self._round_labels:
      labels = {label: tf.round(parsed[label]) for label in self._labels}
    else:
      labels = {label: parsed[label] for label in self._labels}

    return features, labels
