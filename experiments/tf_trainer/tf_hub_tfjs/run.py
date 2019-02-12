"""Experiments with Toxicity Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_trainer.common import base_model
from tf_trainer.common import model_trainer
from tf_trainer.common import serving_input
from tf_trainer.common import tfrecord_input
from tf_trainer.tf_hub_tfjs import model as tf_hub_classifier

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm

FLAGS = tf.app.flags.FLAGS


class TFRecordWithSentencePiece(tfrecord_input.TFRecordInput):
  """Specialized setencepiece based input preprocessor."""

  def __init__(self, spm_path):
    super().__init__()
    self._sp = spm.SentencePieceProcessor()
    self._sp.Load(spm_path)

  def dense_ids(self, texts):
    """Pads sentences ids out to max length, filling with 0's."""
    return pd.DataFrame(
        [self._sp.EncodeAsIds(x) for x in texts]).fillna(0).values.astype(int)

  def pieces(self, feature_dict, label_dict):
    """Processes a batch of texts into sentence pieces."""
    text = feature_dict.pop('text')
    sparse_ids = tf.contrib.layers.dense_to_sparse(
        tf.py_func(self.dense_ids, [text], tf.int64))
    feature_dict['values'] = sparse_ids.values
    feature_dict['indices'] = sparse_ids.indices
    feature_dict['dense_shape'] = sparse_ids.dense_shape
    return feature_dict, label_dict

  def _input_fn_from_file(self, filepath: str):
    filenames_dataset = tf.data.Dataset.list_files(filepath)
    dataset = tf.data.TFRecordDataset(
        filenames_dataset)  # type: tf.data.TFRecordDataset
    # Use parent class parsing to obtain text features, and processed labels.
    parsed_dataset = dataset.map(self._read_tf_example)
    return parsed_dataset.batch(self._batch_size).map(
        self.pieces).prefetch(self._num_prefetch)


def main(argv):
  del argv  # unused

  module = hub.Module(FLAGS.model_spec)
  with tf.Session() as sess:
    spm_path = sess.run(module(signature='spm_path'))

  dataset = TFRecordWithSentencePiece(spm_path)
  model = tf_hub_classifier.TFHubClassifierModel(dataset.labels())

  trainer = model_trainer.ModelTrainer(dataset, model)
  trainer.train_with_eval()

  values = tf.placeholder(tf.int64, shape=[None], name='values')
  indices = tf.placeholder(tf.int64, shape=[None, 2], name='indices')
  dense_shape = tf.placeholder(tf.int64, shape=[None], name='dense_shape')
  serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
    'values': values,
    'indices': indices,
    'dense_shape': dense_shape
  })
  trainer.export(serving_input_fn, None)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
