"""Custom metrics used by Keras models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_trainer import types


def roc_auc(y_true: types.Tensor, y_pred: types.Tensor,
            threshold=0.5) -> types.Tensor:
  """ ROC AUC based on TF's metrics package.

  We assume true labels are 'soft' and pick 0 or 1 based on a threshold.
  """

  y_true = tf.to_int32(y_true > threshold)
  value, update_op = tf.metrics.auc(y_true, y_pred)
  return update_op
