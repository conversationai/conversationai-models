"""Experiments with Toxicity Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_trainer import tfrecord_input
from tf_trainer import text_preprocessor
from tf_trainer import types
from tf_trainer.tf_gru_attention import model

import argparse
import tensorflow as tf

from typing import Dict

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("train_path", "local_data/train.tfrecord",
                           "Path to the training data TFRecord file.")
tf.app.flags.DEFINE_string("validate_path", "local_data/validation.tfrecord",
                           "Path to the validation data TFRecord file.")
tf.app.flags.DEFINE_string("embeddings_path",
                           "local_data/glove.6B/glove.6B.100d.txt",
                           "Path to the embeddings file.")
tf.app.flags.DEFINE_string("text_feature_name", "comment_text",
                           "Feature name of the text feature.")
tf.app.flags.DEFINE_string("model_dir", "/tmp/model_dir",
                           "Directory for the Estimator's model directory.")

# TODO: Missing fields are not handled properly yet.
LABELS = {
    "frac_neg": tf.float32,
    #"frac_very_neg": tf.float32
}  # type: Dict[str, tf.DType]


def main(argv):
  del argv  # unused

  train_path = types.Path(FLAGS.train_path)
  validate_path = types.Path(FLAGS.validate_path)
  embeddings_path = types.Path(FLAGS.embeddings_path)
  text_feature_name = FLAGS.text_feature_name
  model_dir = FLAGS.model_dir

  preprocessor = text_preprocessor.TextPreprocessor(embeddings_path)
  dataset = tfrecord_input.TFRecordInput(
      train_path=train_path,
      validate_path=validate_path,
      text_feature=text_feature_name,
      labels=LABELS,
      word_to_idx=preprocessor.word_to_idx(),
      unknown_token=preprocessor.unknown_token())

  estimator_no_embedding = model.TFRNNModel(
      text_feature_name, LABELS).estimator(
          tf.estimator.RunConfig(model_dir=model_dir))

  estimator = preprocessor.create_estimator_with_embedding(
      estimator_no_embedding, text_feature_name)

  for _ in range(20):
    estimator.train(input_fn=dataset.train_input_fn, steps=1000)
    metrics = estimator.evaluate(input_fn=dataset.validate_input_fn, steps=100)
    tf.logging.info(metrics)


if __name__ == "__main__":
  tf.app.run(main)
