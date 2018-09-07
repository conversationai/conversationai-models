# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import pandas as pd

from dataset import Dataset, Model


class TestModelCompatibleWithInputFn(unittest.TestCase):
  """Tests to verify that the compatibility between input_fn and model."""

  def setUp(self):
    self._model = Model(
        model_name=None, feature_keys=['comment_text'], prediction_keys=[])

  def testCorrect(self):

    def input_fn(max_n_examples, random_filter_keep_rate=1.0):
      assert random_filter_keep_rate
      return pd.DataFrame({
          'comment_text': ['This is one'] * max_n_examples,
          'label_name': [0] * max_n_examples})

    try:
      Dataset(input_fn, self._model)
    except ValueError:
      self.fail('Dataset raised an exception unexpectedly!')

  def testWrongArgInputFn(self):

    def input_fn(max_n_examples, other_args=1.0):
      assert other_args
      return {
          'other_feature': ['This is one'] * max_n_examples,
          'label_name': [0] * max_n_examples}
    with self.assertRaises(Exception) as context:
      Dataset(input_fn, lambda x: x + 1)
      self.assertIn(
          'input_fn should have (at least) `max_n_examples`',
          str(context.exception))

  def testModelWrongType(self):

    def input_fn(max_n_examples, random_filter_keep_rate=1.0):
      assert random_filter_keep_rate
      return {
          'other_feature': ['This is one'] * max_n_examples,
          'label_name': [0] * max_n_examples}

    with self.assertRaises(Exception) as context:
      Dataset(input_fn, lambda x: x + 1)
      self.assertIn(
          'model should be a `Model` instance.', str(context.exception))

  def testInputFnWrongType(self):

    def input_fn(max_n_examples, random_filter_keep_rate=1.0):
      assert random_filter_keep_rate
      return {
          'other_feature': ['This is one'] * max_n_examples,
          'label_name': [0] * max_n_examples}

    with self.assertRaises(Exception) as context:
      Dataset(input_fn, self._model)
      self.assertIn(
          'input_fn should return a pandas DataFrame.',
          str(context.exception))

  def testWrongNumberOfLines(self):

    def input_fn(max_n_examples=1, random_filter_keep_rate=1.0):
      assert random_filter_keep_rate, max_n_examples
      return pd.DataFrame({
          'comment_text': ['This is one'] * 2,
          'label_name': [0] * 2
      })

    with self.assertRaises(Exception) as context:
      Dataset(input_fn, self._model)
      self.assertIn(
          'input_fn(max_n_examples=1) should contain 1 row (exactly).',
          str(context.exception))

  def testInputFnMissingFeatureKeys(self):

    def input_fn(max_n_examples, random_filter_keep_rate=1.0):
      assert random_filter_keep_rate
      return pd.DataFrame(
          {'other_feature': ['This is one'] * max_n_examples,
           'label_name': [0] * max_n_examples})

    with self.assertRaises(Exception) as context:
      Dataset(input_fn, self._model)
      self.assertIn(
          'input_fn must contain at least the feature keys',
          str(context.exception))


if __name__ == '__main__':
  unittest.main()
