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
"""A function to convert csvs to TFRecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("input_csv_path", None,
                           "Path to the input csv file.")
tf.app.flags.DEFINE_string("output_tfrecord_path", None,
                           "Path where the output TFRecord should be written.")
tf.app.flags.DEFINE_string("column_list", None, 
                           "Comma seperated list of column names.")
tf.app.flags.DEFINE_string("dtype_list", None, 
                           "Comma seperated list of column dtypes. "
                           "Each entry should be one of [bytes,str,float,int]).")


def convert_csv_to_tfrecord(input_csv_path,
                            output_tfrecord_path,
                            column_names,
                            column_dtypes):
  df = pd.read_csv(tf.gfile.Open(input_csv_path))
  with tf.python_io.TFRecordWriter(output_tfrecord_path) as writer:
    for row in df.iterrows():
      row = row[1]
      example = tf.train.Example()
      for col_name,dtype in zip(column_names,column_dtypes):
        col_val = row[col_name]
        if dtype == 'bytes':
          example.features.feature[col_name].bytes_list.value.append(
              col_val)
        elif dtype == 'str':
          example.features.feature[col_name].bytes_list.value.append(
              str(col_val).encode("utf-8", errors="replace"))
        elif dtype == 'float':
          example.features.feature[col_name].float_list.value.append(col_val)
        elif dtype == 'int':
          example.features.feature[col_name].int64_list.value.append(col_val)
        else:
          raise ValueError('dtype must be one of bytes, str, float, int.')
      writer.write(example.SerializeToString())


def main(argv):
  del argv  # unused

  input_csv_path = FLAGS.input_csv_path
  output_tfrecord_path = FLAGS.output_tfrecord_path
  column_names = FLAGS.column_list.split(',')
  column_dtypes = FLAGS.dtype_list.split(',')
  assert len(column_names) == len(column_dtypes)

  convert_csv_to_tfrecord(input_csv_path, 
                          output_tfrecord_path,
                          column_names,
                          column_dtypes)


if __name__ == "__main__":
  tf.app.run(main)
