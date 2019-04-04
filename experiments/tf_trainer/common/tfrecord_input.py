"""DatasetInput class based on TFRecord files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import tensorflow as tf
from typing import Callable, List, Dict, Tuple

from tf_trainer.common import base_model
from tf_trainer.common import dataset_input
from tf_trainer.common import types

tf.app.flags.DEFINE_string('train_path', None,
                           'Path to the training data TFRecord file.')
tf.app.flags.DEFINE_string('validate_path', None,
                           'Path to the validation data TFRecord file.')
tf.app.flags.DEFINE_string('labels', 'frac_neg',
                           'Comma separated list of label features.')
tf.app.flags.DEFINE_string(
    'label_dtypes', None, 'Comma separated list of dtypes for labels. Each '
    'dtype must be float or int. If not provided '
    'assumes all labels are floats.')
tf.app.flags.DEFINE_string('text_feature', 'comment_text',
                           'Name of feature containing text input.')
tf.app.flags.DEFINE_boolean('round_labels', True,
                            'Round label features to 0 or 1 if true.')
tf.app.flags.DEFINE_integer('batch_size', 256,
                            'Batch sizes to use when reading.')
tf.app.flags.DEFINE_integer(
  'num_prefetch', 5,
  'An optimization parameter for the number of elements to prefetch. See: '
  'https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch')

FLAGS = tf.app.flags.FLAGS

DTYPE_MAPPING = {'float': tf.float32, 'int': tf.int64}

DTYPE_DEFAULT = {'float': -1.0, 'int': -1}


class TFRecordInput(dataset_input.DatasetInput):
  """Simple no-preprocessing TFRecord based DatasetInput.

  Handles parsing of TF Examples.

  Regardless of which TF Example feature key is used, as specified by the
  FLAGS.text_feature, the simple input will store the input text feature in
  the feature key _text_feature.
  """

  def __init__(self) -> None:
    self._labels = FLAGS.labels.split(',')
    if FLAGS.label_dtypes:
      self._label_dtypes = FLAGS.label_dtypes.split(',')
    else:
      self._label_dtypes = ['float'] * len(self._labels)
    self._batch_size = FLAGS.batch_size
    self._num_prefetch = FLAGS.num_prefetch
    self._text_feature = FLAGS.text_feature
    self._round_labels = FLAGS.round_labels

  def labels(self) -> List[str]:
    """List of the names of the float label features."""
    return self._labels

  def text_feature(self) -> str:
    """Name of the feature containing the input text from examples."""
    return self._text_feature

  def train_input_fn(self) -> types.FeatureAndLabelTensors:
    """input_fn for TF Estimators for training set.

    Automatically repeats over input data forever. We define epoc limits in the
    model trainer.
    """
    assert FLAGS.train_path
    return self._input_fn_from_file(FLAGS.train_path).repeat()

  def validate_input_fn(self) -> types.FeatureAndLabelTensors:
    """input_fn for TF Estimators for validation set."""
    assert FLAGS.validate_path
    return self._input_fn_from_file(FLAGS.validate_path)

  def _keys_to_features(self):
    keys_to_features = {}
    keys_to_features[self._text_feature] = tf.FixedLenFeature([], tf.string)
    for label, dtype in zip(self._labels, self._label_dtypes):
      keys_to_features[label] = tf.FixedLenFeature([], DTYPE_MAPPING[dtype],
                                                   DTYPE_DEFAULT[dtype])
    return keys_to_features

  def _input_fn_from_file(self, filepath: str) -> tf.data.TFRecordDataset:
    filenames_dataset = tf.data.Dataset.list_files(filepath)
    dataset = tf.data.TFRecordDataset(
        filenames_dataset)  # type: tf.data.TFRecordDataset
    parsed_dataset = dataset.map(
        self._read_tf_example, num_parallel_calls=multiprocessing.cpu_count())
    return parsed_dataset.batch(self._batch_size).prefetch(self._num_prefetch)

  def _process_labels(self, features, parsed):
    """Applies rounding and computes weights tied to feature presence.

    For all of the expected labels, if the value is negative, this
    indicates a missing feature from the input. A corresponding
    label name, suffixed by '_weight' will be added to the features
    with a value of 1.0 is present, and 0.0 if absent. The label
    value is rounded up or down (if enabled) and then mapped to
    zero if missing.

    Args:
        features: the input features read from a TF Example.
        parsed: the input labels read from a TF Example.

    Returns:
        A tuple of the features dict (with weights) and the labels dict.
    """
    # Make a deep copy to avoid changing the input.
    new_features = {k: v for k, v in features.items()}
    labels = {}
    for label in self._labels:
      label_value = tf.cast(parsed[label], dtype=tf.float32)
      # Missing values are negative, find them and zero those features out.
      weight = tf.cast(tf.greater_equal(label_value, 0.0), dtype=tf.float32)
      if self._round_labels:
        label_value = tf.round(label_value)
      new_features[label + '_weight'] = weight
      labels[label] = tf.multiply(label_value, weight)
    return new_features, labels

  def _read_tf_example(
      self,
      record: tf.Tensor,
  ) -> types.FeatureAndLabelTensors:
    """Parses TF Example protobuf into a text feature and labels.

    The input TF Example has a text feature as a singleton list with the full
    comment as the single element.
    """
    parsed = tf.parse_single_example(
        record, self._keys_to_features())  # type: Dict[str, types.Tensor]

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
               max_seq_len: int = 30000) -> None:
    super().__init__()
    self._train_preprocess_fn = train_preprocess_fn
    self._max_seq_len = max_seq_len

  def _input_fn_from_file(self, filepath: str) -> types.FeatureAndLabelTensors:

    filenames_dataset = tf.data.Dataset.list_files(filepath)
    dataset = tf.data.TFRecordDataset(
        filenames_dataset)  # type: tf.data.TFRecordDataset

    parsed_dataset = dataset.map(
        self._read_tf_example, num_parallel_calls=multiprocessing.cpu_count())
    parsed_dataset = parsed_dataset.filter(lambda x, _: tf.less(
        x['sequence_length'], self._max_seq_len))

    feature_shapes = {
        base_model.TOKENS_FEATURE_KEY: [None],
        'sequence_length': []
    }
    for label in self._labels:
      feature_shapes[label + '_weight'] = []

    padded_shapes = (
      feature_shapes,
      {label: [] for label in self._labels})  # type: Tuple[Dict, Dict]
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
    parsed = tf.parse_single_example(
        record, self._keys_to_features())  # type: Dict[str, types.Tensor]

    text = parsed[self.text_feature()]
    tokens = self._train_preprocess_fn(text)
    features = {
        base_model.TOKENS_FEATURE_KEY: tokens,
        'sequence_length': tf.shape(tokens)[0],
    }
    return self._process_labels(features, parsed)
