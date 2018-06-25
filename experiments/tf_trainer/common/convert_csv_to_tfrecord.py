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
            features, label = row[0], row[1]
            example = tf.train.Example()
            example.features.feature[COLS[0]].bytes_list.value.append(features.encode('utf-8'))
            example.features.feature[COLS[1]].float_list.value.append(label)
            writer.write(example.SerializeToString())

def main(argv):
  del argv  # unused

  input_csv_path = FLAGS.input_csv_path
  output_tfrecord_path = FLAGS.output_tfrecord_path

  convert_csv_to_tfrecord(input_csv_path, output_tfrecord_path)

if __name__ == "__main__":
  tf.app.run(main)