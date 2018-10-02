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

import getpass
import os
import time
import unittest

from dataset import Dataset
from dataset import Model
import pandas as pd
from utils_tfrecords import EncodingFeatureSpec


class TestCompatibleInputFn(unittest.TestCase):
  """Verifies the compatibility of input_fn with `Dataset`."""

  def testCorrect(self):

    def input_fn(max_n_examples):
      return pd.DataFrame({
          'comment_text': ['This is one'] * max_n_examples,
          'label_name': [0] * max_n_examples
      })

    try:
      Dataset(input_fn, 'dataset_dir')
    except ValueError:
      self.fail('Dataset raised an exception unexpectedly!')

  def testWrongArgInputFn(self):

    def input_fn(other_args=1.0):
      assert other_args
      return {'other_feature': ['This is one'], 'label_name': [0]}

    with self.assertRaises(Exception) as context:
      Dataset(input_fn, 'dataset_dir')
      self.assertIn('input_fn should have (at least) `max_n_examples`',
                    str(context.exception))

  def testInputFnWrongType(self):

    def input_fn(max_n_examples):
      return {
          'other_feature': ['This is one'] * max_n_examples,
          'label_name': [0] * max_n_examples
      }

    with self.assertRaises(Exception) as context:
      Dataset(input_fn, 'dataset_dir')
      self.assertIn('input_fn should return a pandas DataFrame.',
                    str(context.exception))

  def testWrongNumberOfLines(self):

    def input_fn(max_n_examples=1):
      assert max_n_examples
      return pd.DataFrame({
          'comment_text': ['This is one'] * 2,
          'label_name': [0] * 2
      })

    with self.assertRaises(Exception) as context:
      Dataset(input_fn, 'dataset_dir')
      self.assertIn(
          'input_fn(max_n_examples=1) should contain 1 row (exactly).',
          str(context.exception))


class TestModelCompatibleWithInputFn(unittest.TestCase):
  """Verifies the compatibility between input_fn and model."""

  def testBadTypeFeatureKeys(self):

    with self.assertRaises(Exception) as context:
      model = Model(
          feature_keys_spec='comment_text',
          prediction_keys='prediction_key',
          model_names='None',
          project_name=None)
      self.assertIn('Spec should be a dictionary', str(context.exception))

  def testInputFnMissingFeatureKeys(self):

    model = Model(
        feature_keys_spec={'comment_text': EncodingFeatureSpec.LIST_STRING},
        prediction_keys='prediction_key',
        model_names='None',
        project_name=None)

    def input_fn(max_n_examples):
      return pd.DataFrame({
          'other_feature': ['This is one'] * max_n_examples,
          'label_name': [0] * max_n_examples
      })

    with self.assertRaises(Exception) as context:
      dataset = Dataset(input_fn, 'dataset_dir')
      dataset.check_compatibility(model)
      self.assertIn('input_fn must contain at least the feature keys',
                    str(context.exception))

  def testModelIsCompatibleWithDataset(self):
    model = Model(
        feature_keys_spec={'comment_text': EncodingFeatureSpec.LIST_STRING},
        prediction_keys='prediction_key',
        model_names='None',
        project_name=None)

    def input_fn(max_n_examples):
      return pd.DataFrame({
          'comment_text': ['This is one'] * max_n_examples,
          'label_name': [0] * max_n_examples
      })

    try:
      dataset = Dataset(input_fn, 'dataset_dir')
      dataset.check_compatibility(model)
    except ValueError:
      self.fail('Dataset raised an exception unexpectedly!')


class TestEndPipeline(unittest.TestCase):
  """Verifies end-to-end use of dataset."""

  test_version = str(int(time.time()))

  def setUp(self):
    def input_fn_test(max_n_examples):
      return pd.DataFrame({
          'comment_text': [['This', 'is', 'one']] * max_n_examples
      })

    gcs_path_test = os.path.join('gs://kaggle-model-experiments/',
                                 getpass.getuser(), 'unittest', 'dataset_test',
                                 TestEndPipeline.test_version)
    self.dataset = Dataset(input_fn_test, gcs_path_test)
    self.dataset.load_data(5)

    model_input_spec = {
        'comment_text': EncodingFeatureSpec.LIST_STRING,
    }
    self.model = Model(
        feature_keys_spec=model_input_spec,
        prediction_keys='frac_neg/logistic',
        example_key='comment_key',
        model_names=[
            'tf_gru_attention:v_20180914_163804',
            'tf_gru_attention:v_20180823_133625'
        ],
        project_name='wikidetox')

  def testComputePredictions(self):
    try:
      self.dataset.add_model_prediction_to_data(self.model)
    except ValueError :
      self.fail('Dataset raised an exception unexpectedly!')

  def testLoadPredictions(self):
    try:
      self.dataset.add_model_prediction_to_data(
          self.model, recompute_predictions=False)
    except ValueError:
      self.fail('Dataset raised an exception unexpectedly!')


if __name__ == '__main__':
  unittest.main()
