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
"""Tests for tf records utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import unittest

import utils_cloudml


class CallModelPredictionsFromDf(unittest.TestCase):
  """Tests for `call_model_predictions_from_df`."""

  #TODO(fprost): Implement these.

  def test_correct(self):
    return


class CheckJobOver(unittest.TestCase):
  """Tests for `check_job_over`."""

  # TODO(fprost): Implement these.
  def test_correct(self):
    return


class AddModelPredictionsToDf(unittest.TestCase):
  """Tests for `add_model_predictions_to_df`."""

  def setUp(self):
    self.COMMENT_KEY = 'comment_key'
    self._df = pd.DataFrame({
        self.COMMENT_KEY: [0, 1],
        'other_field_1': ['I am a man', 'I am a woman'],
        })
    self._prediction_file = 'gs://kaggle-model-experiments/files_for_unittest/model1:v1'
    self._model_col_name = 'model1:v1_preds'
    self._prediction_name = 'toxicity/logistic'
    self._example_key = self.COMMENT_KEY

  def test_missing_prediction_file(self):
    path = 'not_existing_folder/not_existing_file_path'

    with self.assertRaises(Exception) as context:
      utils_cloudml.add_model_predictions_to_df(
          self._df,
          path,
          self._model_col_name,
          self._prediction_name,
          self._example_key)
      self.assertIn(
          'Prediction file does not exist.',
          str(context.exception))

  def test_empty_prediction_file(self):
    path = 'gs://kaggle-model-experiments/files_for_unittest/for_empty_predictions'

    with self.assertRaises(Exception) as context:
      utils_cloudml.add_model_predictions_to_df(
          self._df,
          path,
          self._model_col_name,
          self._prediction_name,
          self._example_key)
    self.assertIn(
        'The prediction file returned by CMLE is empty.',
        str(context.exception))

  def test_missing_example_key(self):
    example_key = 'not_found_example_key'
    with self.assertRaises(Exception) as context:
      utils_cloudml.add_model_predictions_to_df(
          self._df,
          self._prediction_file,
          self._model_col_name,
          self._prediction_name,
          example_key,
          )
    self.assertIn(
        "Predictions do not contain the 'example_key' field.",
        str(context.exception))

  def test_missing_prediction_key(self):
    prediction_key = 'not_found_prediction_key'
    with self.assertRaises(Exception) as context:
      utils_cloudml.add_model_predictions_to_df(
          self._df,
          self._prediction_file,
          self._model_col_name,
          prediction_key,
          self._example_key)
    self.assertIn(
        "Predictions do not contain the 'prediction_name' field.",
        str(context.exception))

  def test_correct(self):
    output_df = utils_cloudml.add_model_predictions_to_df(
        self._df,
        self._prediction_file,
        self._model_col_name,
        self._prediction_name,
        self._example_key)
    right_output = pd.DataFrame({
        self.COMMENT_KEY: [0, 1],
        'other_field_1': ['I am a man', 'I am a woman'],
        self._model_col_name: [0.38753455877304077, 0.045782867819070816]
        })
    pd.testing.assert_frame_equal(
        output_df.sort_index(axis=1), right_output.sort_index(axis=1))


if __name__ == '__main__':
  unittest.main()