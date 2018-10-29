"""Experiments with Toxicity Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

# Import common flags and run code. Must be imported first.
from tf_trainer.common import model_trainer

from tf_trainer.common import dataset_input
from tf_trainer.common import serving_input
from tf_trainer.common import types
from tf_trainer.tf_hub_classifier import model as tf_hub_classifier

import tensorflow as tf

from typing import Dict

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("text_feature_name", "comment_text",
                           "Feature name of the text feature.")
tf.app.flags.DEFINE_string("key_name", "comment_key",
                           "Name of the key feature for serving examples.")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "The batch size to use during training.")
tf.app.flags.DEFINE_integer("train_steps", 40000,
                            "The number of steps to train for.")
tf.app.flags.DEFINE_integer("eval_period", 500,
                            "The number of steps per eval period.")
tf.app.flags.DEFINE_integer("eval_steps", 50,
                            "The number of steps to eval for.")

# TODO: Missing fields are not handled properly yet.
LABELS = {
    "frac_neg": tf.float32,
    #"frac_very_neg": tf.float32
}  # type: Dict[str, tf.DType]


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
        self._read_tf_example, num_parallel_calls=multiprocessing.cpu_count())

    padded_shapes = ({
        self._text_feature: [],
    }, {label: [] for label in self._labels})
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

    features = {self._text_feature: tf.convert_to_tensor([parsed[self._text_feature]])}
    if self._round_labels:
      labels = {label: [tf.round(parsed[label])] for label in self._labels}
    else:
      labels = {label: [parsed[label]] for label in self._labels}
    print(features)
    return features, labels


def create_serving_input_fn(text_feature_name, key_name):

  def serving_input_fn_tfrecords():

    serialized_example = tf.placeholder(
        shape=[],
        dtype=tf.string,
        name="input_example_tensor"
    )
    feature_spec = {
        text_feature_name: tf.FixedLenFeature([], dtype=tf.string),
        key_name: tf.FixedLenFeature([], dtype=tf.int64)
    }

    features = tf.parse_example(
        serialized_example, feature_spec)

    return tf.estimator.export.ServingInputReceiver(
        features,
        serialized_example)

  return serving_input_fn_tfrecords


def main(argv):
  del argv  # unused

  dataset = TFSimpleRecordInput(
      train_path=FLAGS.train_path,
      validate_path=FLAGS.validate_path,
      text_feature=FLAGS.text_feature_name,
      labels=LABELS,
      batch_size=FLAGS.batch_size)

  model = tf_hub_classifier.TFRNNModel(
      FLAGS.text_feature_name,
      set(LABELS.keys())
      )

  trainer = model_trainer.ModelTrainer(dataset, model)
  trainer.train_with_eval(FLAGS.train_steps, FLAGS.eval_period, FLAGS.eval_steps)

  serving_input_fn = create_serving_input_fn(
      text_feature_name=FLAGS.text_feature_name,
      key_name=FLAGS.key_name)
  trainer.export(serving_input_fn)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
