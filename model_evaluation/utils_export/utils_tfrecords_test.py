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

import unittest

import pandas as pd
import tensorflow as tf

import utils_tfrecords


class TestEncodingAndDecoding(unittest.TestCase):
  """Test to encode and decode a pandas DataFrame"""

  def testCorrect(self):
    input_df = pd.DataFrame({
        'x': [1, 2, 3],
        'y': ['a', 'b', 'c'],
        'z': [['a', 'b'], ['c', 'd'], ['e', 'f']]
        })
    encoding_feature_spec = {
        'x': utils_tfrecords.EncodingFeatureSpec.INTEGER,
        'y': utils_tfrecords.EncodingFeatureSpec.STRING,
        'z': utils_tfrecords.EncodingFeatureSpec.LIST_STRING
        }
    decoding_spec = {
        'x': tf.FixedLenFeature([], dtype=tf.int64),
        'y': tf.FixedLenFeature([], dtype=tf.string),
        'z': tf.FixedLenFeature([2], dtype=tf.string),
        }
    tf_records_path = 'unittest.tf_records'
    utils_tfrecords.encode_pandas_to_tfrecords(input_df, encoding_feature_spec, tf_records_path)

    output_df = utils_tfrecords.decode_tf_records_to_pandas(
        decoding_spec,
        tf_records_path)
    try:
      pd.testing.assert_frame_equal(input_df, output_df)
    except ValueError:
      self.fail('Dataset raised an exception unexpectedly!')


class TestFeatureKeySpec(unittest.TestCase):
  """Verifies the format of Feature Spec"""

  def test_not_a_dictionary(self):
    feature_keys_spec = 'not_a_dict', 
    with self.assertRaises(Exception) as context:
      utils_tfrecords.is_valid_spec(feature_keys_spec)
    self.assertIn(
        'Spec should be a dictionary instance.',
        str(context.exception))

  def test_not_in_possible(self):
    feature_keys_spec = {'key': 'other_possibility'} 
    with self.assertRaises(Exception) as context:
      utils_tfrecords.is_valid_spec(feature_keys_spec)
    self.assertIn(
        'Spec is badly defined. Authorized types are one of',
        str(context.exception))

  def test_valid(self):
    try:
      feature_keys_spec = {'comment_text': utils_tfrecords.EncodingFeatureSpec.LIST_STRING}
      utils_tfrecords.is_valid_spec(feature_keys_spec)
    except ValueError:
      self.fail('Dataset raised an exception unexpectedly!')


if __name__ == '__main__':
  unittest.main()
