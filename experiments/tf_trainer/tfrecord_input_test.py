"""Tests for tfrecord_input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_trainer import tfrecord_input
from tf_trainer import text_preprocessor
from tf_trainer import types

import numpy as np
import tensorflow as tf


class TFRecordInputTest(tf.test.TestCase):

  def test_read_tf_example(self):
    ex = tf.train.Example(
        features=tf.train.Features(
            feature={
                "label":
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=[0.8])),
                "comment":
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=["Hi there.".encode("utf-8")]))
            }))
    ex_tensor = tf.convert_to_tensor(ex.SerializeToString(), dtype=tf.string)

    dataset_input = tfrecord_input.TFRecordInput(
        train_path=None,
        validate_path=None,
        text_feature="comment",
        labels={"label": tf.float32},
        word_to_idx={
            "Hi": 12,
            "there": 13
        },
        unknown_token=999)

    with self.test_session():
      features, labels = dataset_input._read_tf_example(ex_tensor)
      self.assertEqual(list(features["comment"].eval()), [12, 13, 999])
      self.assertAlmostEqual(labels["label"].eval(), 0.8)


if __name__ == "__main__":
  tf.test.main()
