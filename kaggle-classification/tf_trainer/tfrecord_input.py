"""DatasetInput class based on TFRecord files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import functools
from tf_trainer import dataset_input
from tf_trainer import types
from typing import Dict, Set, Tuple, Callable


class TFRecordInput(dataset_input.DatasetInput):
  """TFRecord based DatasetInput."""

  # TODO: Parameterize
  LABELS = set([
      "frac_neg", "frac_very_neg", "obscene", "threat", "insult",
      "identity_hate"
  ])  # type: Set[str]

  def __init__(self, train_file: types.Path, validate_file: types.Path) -> None:
    self._train_file = train_file  # type: types.Path
    self._validate_file = validate_file  # type: types.Path

  def train_input_fn(self) -> Callable[[], types.FeatureAndLabelTensors]:
    return functools.partial(TFRecordInput.input_fn_from_file, self._train_file)

  def validate_input_fn(self) -> Callable[[], types.FeatureAndLabelTensors]:
    return functools.partial(TFRecordInput.input_fn_from_file,
                             self._validate_file)

  @staticmethod
  def input_fn_from_file(
      filename: types.Path) -> Tuple[types.Tensor, types.TensorDict]:
    dataset = tf.data.TFRecordDataset(filename)  # type: tf.data.TFRecordDataset

    def parser(
        record: tf.Tensor) -> Tuple[types.Tensor, Dict[str, types.Tensor]]:
      # TODO: Consider adding defaults
      keys_to_features = {
          # Parameterize
          "comment_text": tf.FixedLenFeature([], tf.string),
      }
      for label in TFRecordInput.LABELS:
        keys_to_features[label] = tf.FixedLenFeature([], tf.float32)
      parsed = tf.parse_single_example(
          record, keys_to_features)  # type: Dict[str, types.Tensor]

      comment_text = parsed["comment_text"]
      labels = {label: parsed[label] for label in TFRecordInput.LABELS}

      return comment_text, labels

    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(128)
    dataset = dataset.repeat(3)  # Num epochs
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    return features, labels
