'''Serving functions for deployed model.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops


FLAGS = tf.app.flags.FLAGS

def create_serving_input_fn(word_to_idx, unknown_token, text_feature_name):

  def serving_input_fn_tfrecords():

    serialized_example = tf.placeholder(
        shape=[None],
        dtype=tf.string,
        name="input_example_tensor"
    )
    feature_spec = {
        text_feature_name: tf.VarLenFeature(dtype=tf.string),
        FLAGS.key_name: tf.FixedLenFeature([], dtype=tf.int64,
                                           default_value=-1)
    }

    features = tf.parse_example(
        serialized_example, feature_spec)

    keys = list(word_to_idx.keys())
    values = list(word_to_idx.values())
    vocabulary_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            keys,
            values,
            key_dtype=tf.string,
            value_dtype=tf.int64),
        unknown_token)
    words_int_sparse = vocabulary_table.lookup(features[text_feature_name])
    words_int_dense = tf.sparse_tensor_to_dense(
        words_int_sparse,
        default_value=0)
    features[text_feature_name] = words_int_dense

    return tf.estimator.export.ServingInputReceiver(
        features,
        serialized_example)

  return serving_input_fn_tfrecords
