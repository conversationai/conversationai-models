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

"""Tests for tfrecord_simple input class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_trainer.common import base_model
from tf_trainer.common import tfrecord_simple
from tf_trainer.common import types

import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

class TFSImpleRecordInputTest(tf.test.TestCase):

  def setUp(self):
    FLAGS.text_feature = 'comment'
    ex = tf.train.Example(
      features=tf.train.Features(
          feature={
              'label':
                  tf.train.Feature(
                      float_list=tf.train.FloatList(value=[0.8])),
              'ignored-label':
                  tf.train.Feature(
                      float_list=tf.train.FloatList(value=[0.125])),
              'comment':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=['Hi there Bob'.encode('utf-8')]))
                  }))
    self.ex_tensor = tf.convert_to_tensor(ex.SerializeToString(),
                                          dtype=tf.string)

  def test_TFSimpleRecordInput_unrounded(self):
    FLAGS.round_labels = False
    FLAGS.labels = 'label'
    dataset_input = tfrecord_simple.TFSimpleRecordInput()

    with self.test_session():
      features, labels = dataset_input._read_tf_example(self.ex_tensor)
      self.assertEqual(features[base_model.TEXT_FEATURE_KEY].eval(),
                       b'Hi there Bob')
      np.testing.assert_almost_equal(labels['label'].eval(), 0.8)
      self.assertEqual(list(labels), ['label'])

  def test_TFSimpleRecordInput_default_values(self):
    FLAGS.labels = 'label,fake_label'
    FLAGS.round_labels = False
    dataset_input = tfrecord_simple.TFSimpleRecordInput()

    with self.test_session():
      features, labels = dataset_input._read_tf_example(self.ex_tensor)
      self.assertEqual(features[base_model.TEXT_FEATURE_KEY].eval(),
                       b'Hi there Bob')
      np.testing.assert_almost_equal(labels['label'].eval(), 0.8)
      np.testing.assert_almost_equal(labels['fake_label'].eval(), -1.0)

  def test_TFSimpleRecordInput_rounded(self):
    FLAGS.labels = 'label'
    FLAGS.round_labels = True
    dataset_input = tfrecord_simple.TFSimpleRecordInput()

    with self.test_session():
      features, labels = dataset_input._read_tf_example(self.ex_tensor)
      self.assertEqual(features[base_model.TEXT_FEATURE_KEY].eval(),
                       b'Hi there Bob')
      np.testing.assert_almost_equal(labels['label'].eval(), 1.0)

if __name__ == '__main__':
  tf.test.main()
