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
"""Defines some utilities to use TF-Records with pandas DataFrame."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import random
import re

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_list_feature(value_list):
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(
          value=[tf.compat.as_bytes(value) for value in value_list]))


class EncodingFeatureSpec(object):

  INTEGER = 'integer'
  STRING = 'string'
  LIST_STRING = 'list_string'

  CONSTRUCTOR_PER_TYPE = {
      INTEGER: _int64_feature,
      STRING: _bytes_feature,
      LIST_STRING: _bytes_list_feature
  }


def is_valid_spec(spec):
  """Verfies that the spec matches requirements."""
  if not isinstance(spec, dict):
    raise ValueError('Spec should be a dictionary instance.')
  for (key, item) in spec.items():
    if not isinstance(key, str):
      raise ValueError(
          'Spec is badly defined. Keys should be string (field names).')
    if item not in EncodingFeatureSpec.CONSTRUCTOR_PER_TYPE.keys():
      raise ValueError(
          'Spec is badly defined. Authorized types are one of {}.'.format(
              EncodingFeatureSpec.CONSTRUCTOR_PER_TYPE.keys()))


def encode_pandas_to_tfrecords(df,
                               feature_keys_spec,
                               tf_records_path,
                               example_key=None):
  """Write a pandas `DataFrame` to a tf_record.

  Args:
    df: pandas `DataFrame`. It must include the fields that are part of
      feature_key_spec.
    feature_keys_spec: Dict of {name: type}, which describes the spec of the
      TF-records.
    tf_records_path: where to write the tf records.
    example_key: key identifier of an example (string). This key will be added
      to data automatically and should not be part of df. If none, no
      example_key will be created.

  Raises:
    ValueError if feature_keys_spec does not follow a FeatureSpec format.

  Note: TFRecords will have fields feature_keys_spec and
  `example_key`(optional).
  """

  is_valid_spec(feature_keys_spec)

  writer = tf.python_io.TFRecordWriter(tf_records_path)
  for i in range(len(df)):

    if not i % 10000:
      logging.info('Preparing train data: {}/{}'.format(i, len(df)))

    # Create a feature
    feature_dict = {}
    for feature in feature_keys_spec:
      constructor = EncodingFeatureSpec.CONSTRUCTOR_PER_TYPE[
          feature_keys_spec[feature]]
      feature_dict[feature] = constructor(df[feature].iloc[i])
      if example_key:
        feature_dict[example_key] = _int64_feature(i)
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

  writer.close()


def decode_tf_records_to_pandas(decoding_features_spec,
                                tf_records_path,
                                max_n_examples=None,
                                random_filter_keep_rate=1.0,
                                filter_fn=None):
  """Loads tf-records into a pandas dataframe.

  Args:
    decoding_features_spec: A dict mapping feature keys to FixedLenFeature
      values. Spec of the tf-records.
    tf_records_path: path to the file
    max_n_examples: Maximum number of examples to extract.
    random_filter_keep_rate: Probability for each line to be kept in training
      data. For each line, we generate a random number x and keep it if x <
      random_filter_keep_rate.
    filter_fn (optional): Function applied to an example. If it returns False,
      the example will be discarded.

  Returns:
    A pandas `DataFrame`.
  """

  if not max_n_examples:
    max_n_examples = float('inf')

  reader = tf.TFRecordReader()
  filenames = tf.train.match_filenames_once(tf_records_path)
  filename_queue = tf.train.string_input_producer(filenames,
                                                  num_epochs=1)

  _, serialized_example = reader.read(filename_queue)
  read_data = tf.parse_single_example(
      serialized=serialized_example, features=decoding_features_spec)

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  sess.run(tf.tables_initializer())
  tf.train.start_queue_runners(sess)

  d = []
  new_line = sess.run(read_data)
  count = 0
  while new_line:
    if filter_fn:
      keep_line = filter_fn(new_line)
    else:
      keep_line = True
    keep_line = keep_line and (random.random() < random_filter_keep_rate)

    if keep_line:
      d.append(new_line)
      count += 1
      if count >= max_n_examples:
        break
      if not (count % 100000):
        logging.info('Loaded {} lines.'.format(count))

    try:
      new_line = sess.run(read_data)
    except tf.errors.OutOfRangeError:
      logging.info('End of file.')
      break

  res = pd.DataFrame(d)
  return res
