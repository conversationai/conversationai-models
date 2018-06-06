"""DatasetInput class based on TFRecord files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import nltk
import functools
import numpy as np
from tf_trainer import dataset_input
from tf_trainer import types
from typing import Dict, Tuple
from tf_trainer import types


class TFRecordInput(dataset_input.DatasetInput):
  """TFRecord based DatasetInput.

  Handles parsing of TF Examples and tokenizing text (with nltk). Note that
  tokenization is currently implemented with tf.py_func and not with tensorflow
  ops.
  """

  def __init__(self,
               train_path: types.Path,
               validate_path: types.Path,
               text_feature: str,
               labels: Dict[str, tf.DType],
               word_to_idx: Dict[str, int],
               unknown_token: int,
               batch_size: int = 32,
               max_seq_length: int = 300) -> None:
    nltk.download('punkt')
    self._train_path = train_path  # type: types.Path
    self._validate_path = validate_path  # type: types.Path
    self._text_feature = text_feature  # type: str
    self._labels = labels  # type: Dict[str, tf.Dtype]
    self._batch_size = batch_size  # type: int
    self._max_seq_length = max_seq_length  # type: int
    self._word_to_idx = word_to_idx  # type: Dict[str, int]
    self._unknown_token = unknown_token  # type: int

  def train_input_fn(self) -> types.FeatureAndLabelTensors:
    """input_fn for TF Estimators for training set."""
    return self._input_fn_from_file(self._train_path)

  def validate_input_fn(self) -> types.FeatureAndLabelTensors:
    """input_fn for TF Estimators for validation set."""
    return self._input_fn_from_file(self._validate_path)

  def _input_fn_from_file(self,
                          filepath: types.Path) -> tf.data.TFRecordDataset:
    dataset = tf.data.TFRecordDataset(filepath)  # type: tf.data.TFRecordDataset
    text_feature = self._text_feature

    def readTFExample(record: tf.Tensor) -> types.FeatureAndLabelTensors:
      """Parses TF Example protobuf into a text feature and labels.

      The input TF Example has a text feature as a singleton list with the full
      comment as the single element.
      """

      keys_to_features = {}
      keys_to_features[text_feature] = tf.FixedLenFeature([], tf.string)
      for label, dtype in self._labels.items():
        keys_to_features[label] = tf.FixedLenFeature([], dtype)
      parsed = tf.parse_single_example(
          record, keys_to_features)  # type: Dict[str, types.Tensor]

      text = parsed[text_feature]
      tokenized_text = tf.py_func(self._tokenize, [text], tf.int64)
      features = {text_feature: tokenized_text}
      labels = {label: parsed[label] for label in self._labels}

      return features, labels

    parsed_dataset = dataset.map(readTFExample)
    batched_dataset = parsed_dataset.padded_batch(
        self._batch_size,
        padded_shapes=(
            {
                # TODO: truncate to max_seq_length
                text_feature: [None]
            },
            {label: [] for label in self._labels}))

    itr = batched_dataset.make_one_shot_iterator().get_next()
    return itr

  def _tokenize(self, text: bytes) -> np.ndarray:
    # IMPORTANT: After tokenization we need to re-encode the text or there will
    # be errors relating to unicode characters.
    return np.asarray([
        self._word_to_idx.get(w, self._unknown_token)
        for w in nltk.word_tokenize(text.decode('utf-8'))
    ])
