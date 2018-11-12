# coding=utf-8
# Copyright 2018 The Conversation-AI.github.io Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for BaseKerasModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
from tf_trainer.common import types
from tf_trainer.common import base_keras_model

import numpy as np
import tensorflow as tf


class BaseKerasModelTest(tf.test.TestCase):

  @contextmanager
  def initialized_test_session(self):
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_g)
      sess.run(init_l)
      yield sess

  def test_roc_auc(self):
      y_true = tf.convert_to_tensor([0.0, 0.7, 0.3, 1.0, 0.2, 1.0, 0.0], dtype=tf.float32)
      # Note that because use of underlying tf.metrics.auc, which uses
      # trapezoidal summation by default, AUC computations are not really
      # symetric.
      #
      # An example where everything is opposite of what it should be.
      y_pred_010_auc = tf.convert_to_tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=tf.float32)
      # 4/7 examples are correct, results in 0.5
      y_pred_050_auc = tf.convert_to_tensor([0.2, 0.8, 0.2, 0.9, 1.0, 0.1, 0.5], dtype=tf.float32)
      # 5/7 examples are correct, results in 0.8
      y_pred_080_auc = tf.convert_to_tensor([0.2, 0.8, 0.2, 0.9, 1.0, 1.0, 0.5], dtype=tf.float32)
      # 7/7 examples are correct, results in 0.999....
      y_pred_100_auc = tf.convert_to_tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=tf.float32)

      auc_010 = base_keras_model.BaseKerasModel.roc_auc(y_true, y_pred_010_auc, threshold=0.5)
      auc_050 = base_keras_model.BaseKerasModel.roc_auc(y_true, y_pred_050_auc, threshold=0.5)
      auc_080 = base_keras_model.BaseKerasModel.roc_auc(y_true, y_pred_080_auc, threshold=0.5)
      auc_100 = base_keras_model.BaseKerasModel.roc_auc(y_true, y_pred_100_auc, threshold=0.5)

      with self.initialized_test_session():
        self.assertAlmostEqual(auc_010.eval(), 0.1, 1)
        self.assertAlmostEqual(auc_050.eval(), 0.5, 1)
        self.assertAlmostEqual(auc_080.eval(), 0.8, 1)
        self.assertAlmostEqual(auc_100.eval(), 1.0, 1)


if __name__ == "__main__":
  tf.test.main()
