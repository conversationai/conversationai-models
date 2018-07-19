'''Serving functions for deployed model.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("serving_format", "TFRECORDS",
                           "Format of inputs in inference."
                           "Can be either JSON or TFRECORDS.")


def create_serving_input_fn(feature_preprocessor_init, text_feature_name, key_name):

  def serving_input_fn_json():
    features_placeholders = {}
    features_placeholders[text_feature_name] = array_ops.placeholder(
        dtype=tf.string, name=text_feature_name)
    features_placeholders[key_name] = array_ops.placeholder(
        dtype=tf.string, name=key_name)

    features = {}
    features[key_name] = features_placeholders[key_name]
    feature_preprocessor = feature_preprocessor_init()
    features[text_feature_name] = feature_preprocessor(
      features_placeholders[text_feature_name])

    return tf.estimator.export.ServingInputReceiver(
        features,
        features_placeholders)

  def serving_input_fn_tfrecords():
    serialized_example = tf.placeholder(
        shape=[None],
        dtype=tf.string,
        name="input_example_tensor"
    )
    feature_spec = {
        text_feature_name: tf.FixedLenFeature([], dtype=tf.string),
        key_name: tf.FixedLenFeature([], dtype=tf.int64)
    }

    features = tf.parse_example(
        serialized_example, feature_spec)
    feature_preprocessor = feature_preprocessor_init()
    features[text_feature_name] = feature_preprocessor(
        features[text_feature_name])

    return tf.estimator.export.ServingInputReceiver(
        features,
        serialized_example)
  
  if FLAGS.serving_format == 'TFRECORDS':
    return serving_input_fn_tfrecords
  elif FLAGS.serving_format == 'JSON':
    return serving_input_fn_json
  else:
    raise ValueError('Serving format not implemented.'
        ' Should be one of ["JSON", "TFRECORDS"].'
        )
