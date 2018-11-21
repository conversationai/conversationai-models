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

COLS = ['comment_text', 'frac_neg']

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("input_csv_path",
                           None,
                           "Path to the input csv file.")
tf.app.flags.DEFINE_string("output_tfrecord_path", None,
                           "Path where the output TFRecord should be written.")


def convert_csv_to_tfrecord(input_csv_path, output_tfrecord_path):
    df = pd.read_csv(tf.gfile.Open(input_csv_path))
    csv = df[COLS].values
    with tf.python_io.TFRecordWriter(output_tfrecord_path) as writer:
        for row in csv:
            text, label = row[0], row[1]
            example = tf.train.Example()
            example.features.feature[COLS[0]].bytes_list.value.append(text.encode('utf-8', errors='replace'))
            example.features.feature[COLS[1]].float_list.value.append(label)
            writer.write(example.SerializeToString())

def main(argv):
  del argv  # unused

  input_csv_path = FLAGS.input_csv_path
  output_tfrecord_path = FLAGS.output_tfrecord_path

  convert_csv_to_tfrecord(input_csv_path, output_tfrecord_path)

if __name__ == "__main__":
  tf.app.run(main)
