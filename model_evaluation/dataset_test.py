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

import pandas as pd
import unittest

from dataset import Dataset, Model

class TestModelCompatibleWithInputFn(unittest.TestCase):
  """Tests to verify that the compatibility between input_fn and model.""" 
  
  def setUp(self):
    self._model = Model(
        model_name=None,
        feature_keys=['comment_text'],
        prediction_keys=[])

  def testCorrect(self):  
    def input_fn(max_n_examples, random_filter_keep_rate=1.0):
      return pd.DataFrame({'comment_text': ['This is one'], 'label_name': [0]})
    try:
      Dataset(input_fn, self._model)
    except:
      self.fail("Dataset raised an exception unexpectedly!")

  def testInputFnWrongType(self):
    def input_fn(max_n_examples, random_filter_keep_rate=1.0):
      return {'other_feature': ['This is one'], 'label_name': [0]}
    with self.assertRaises(Exception) as context:
      Dataset(input_fn, self._model)
      self.assertTrue(
          'input_fn should return a pandas DataFrame.' in str(context.exception))

  def testWrongNumberOfLines(self):
    def input_fn(max_n_examples, random_filter_keep_rate=1.0):
      return pd.DataFrame({'comment_text': ['This is one']*2, 'label_name': [0]*2})
    with self.assertRaises(Exception) as context:
      Dataset(input_fn, self._model)
      self.assertTrue(
          'input_fn(max_n_examples=1) should contain 1 row (exactly).' in str(context.exception))

  def testInputFnMissingFeatureKeys(self):
    def input_fn(max_n_examples, random_filter_keep_rate=1.0):
      return pd.DataFrame({'other_feature': ['This is one'], 'label_name': [0]})
    with self.assertRaises(Exception) as context:
      Dataset(input_fn, self._model)
      self.assertTrue(
          'input_fn must contain at least the feature keys' in str(context.exception))


if __name__ == '__main__':
  unittest.main()