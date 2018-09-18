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
"""Tests for cloud MLE utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import pandas as pd

import utils_export.utils_cloudml


class TestFeatureKeySpec(unittest.TestCase):
  """Verifies that the compatibility between input_fn and model."""

  def test_not_a_dictionary(self):
    feature_keys_spec = 'not_a_dict', 
    with self.assertRaises(Exception) as context:
      utils_cloudml.is_valid_spec(feature_keys_spec)
      self.assertIn(
          'Spec should be a dictionary instance.',
          str(context.exception))

  def test_not_in_possible(self):
    feature_keys_spec = {'key': 'other_possibility'} 
    with self.assertRaises(Exception) as context:
      utils_cloudml.is_valid_spec(feature_keys_spec)
      self.assertIn(
          'Spec is badly defined. Authorized types are one of',
          str(context.exception))

  def test_valid(self):
    try:
      feature_keys_spec = {'comment_text': utils_cloudml.FeatureSpec.STRINGLIST}
      utils_cloudml.is_valid_spec(feature_keys_spec)
    except ValueError:
      self.fail('Dataset raised an exception unexpectedly!')

if __name__ == '__main__':
  unittest.main()
