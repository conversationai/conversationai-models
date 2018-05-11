"""TODO(jjtan): DO NOT SUBMIT without one-line documentation for batch_generator.

TODO(jjtan): DO NOT SUBMIT without a detailed description of batch_generator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections.abc import Generator
import tensorflow as tf


class TFRecordBatchGenerator(Generator):
  """Batch Generator
  """

  def __init__(self, tfrecord_file):
    dataset = tf.data.TFRecordDataset(filenames=tfrecord_file)
    dataset = dataset.map(self._parse_function)
    dataset = dataset.batch(4)  # Batch size to use
    self.example_itr = dataset.make_one_shot_iterator()

  def send(self, ignored_arg):
    examples = self.example_itr.get_next()
    return examples

  def throw(self):
    raise StopIteration

  def _parse_function(self, serialized):
    features = {
        'comment_text': tf.FixedLenFeature([], tf.string),
        'frac_neg': tf.FixedLenFeature([], tf.float32),  # toxic
        'frac_very_neg': tf.FixedLenFeature([], tf.float32),  # severe_toxic
        'obscene': tf.FixedLenFeature([], tf.float32),
        'threat': tf.FixedLenFeature([], tf.float32),
        'insult': tf.FixedLenFeature([], tf.float32),
        'identity_hate': tf.FixedLenFeature([], tf.float32)
    }
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(
        serialized=serialized, features=features)
    return parsed_example
