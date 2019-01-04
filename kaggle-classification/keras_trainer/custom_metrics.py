"""Custom metrics used by Keras models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def auc_roc(y_true, y_pred):
  # any tensorflow metric
  y_true = tf.to_int32(tf.greater(y_true, 0.5))
  value, update_op = tf.metrics.auc(y_true, y_pred)

  # find all variables created for this metric
  metric_vars = [
      i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]
  ]

  # Add metric variables to GLOBAL_VARIABLES collection.
  # They will be initialized for new session.
  for v in metric_vars:
    tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

  # force update metric values
  with tf.control_dependencies([update_op]):
    value = tf.identity(value)
    return value
