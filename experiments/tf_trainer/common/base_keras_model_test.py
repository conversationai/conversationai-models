"""Tests for BaseKerasModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_trainer.common import types
from tf_trainer.common import base_keras_model

import numpy as np
import tensorflow as tf


class BaseKerasModelTest(tf.test.TestCase):

  def test_roc_auc(self):
    y_true = tf.convert_to_tensor([0.0, 0.7, 0.3, 1.0, 0.2, 1.0, 0.0], dtype=tf.float32)
    y_pred_05_auc = tf.convert_to_tensor([0.2, 0.8, 0.2, 0.9, 1.0, 0.1, 0.5], dtype=tf.float32)
    y_pred_08_auc = tf.convert_to_tensor([0.2, 0.8, 0.2, 0.9, 1.0, 1.0, 0.5], dtype=tf.float32)

    auc_05 = base_keras_model.BaseKerasModel.roc_auc(y_true, y_pred_05_auc, threshold=0.5)
    auc_08 = base_keras_model.BaseKerasModel.roc_auc(y_true, y_pred_08_auc, threshold=0.5)

    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_g)
      sess.run(init_l)
      self.assertAlmostEqual(auc_05.eval(), 0.5, 1)
      self.assertAlmostEqual(auc_08.eval(), 0.8, 1)


if __name__ == "__main__":
  tf.test.main()
