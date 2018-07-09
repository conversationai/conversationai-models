'''Serving functions for deployed model.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops

def create_serving_input_fn(feature_preprocessor_init, sentence_name, key_name):

  def serving_input_fn():
    features_placeholders = {}
    features_placeholders[sentence_name] = array_ops.placeholder(
        dtype=tf.string, name=sentence_name)
    features_placeholders[key_name] = array_ops.placeholder(
        dtype=tf.string, name=key_name)

    features = features_placeholders
    feature_preprocessor = feature_preprocessor_init()
    features[sentence_name] = feature_preprocessor(
      features_placeholders[sentence_name]) 

    return tf.estimator.export.ServingInputReceiver(
        features,
        features_placeholders)

  return serving_input_fn
