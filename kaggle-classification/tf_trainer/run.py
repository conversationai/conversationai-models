"""Experiments with Toxicity Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from tf_trainer import tfrecord_input
from tf_trainer import text_preprocessor
from tf_trainer import keras_rnn_model
from tf_trainer import types

import argparse
import tensorflow as tf

from typing import Dict

FEATURE = "comment_text"  # type: str
# TODO: Missing fields are not handled properly yet.
LABELS = {
    "frac_neg": tf.float32,
}  # type: Dict[str, tf.DType]


def add_embedding_to_estimator(
    estimator: tf.estimator.Estimator,
    text_feature_name: str,
    text_preprocessor: text_preprocessor.TextPreprocessor,
    trainable: bool = False) -> tf.estimator.Estimator:
  """Takes an existing estimator and prepends the embedding layers to it.

  Args:
    estimator: A predefined Estimator that expects embeddings.
    text_feature_name: The name of the feature containing the text.
    text_preprocess: An instance of TextPreprocessor holding embedding info.
    trainable: If we want to update the embedding.

  Returns:
    TF Estimator with embedding ops added.
  """
  old_model_fn = estimator.model_fn
  old_config = estimator.config
  old_params = estimator.params

  def new_model_fn(features, labels, mode, params, config):
    """model_fn used in defining the new TF Estimator"""

    word_to_idx_table = text_preprocessor.word_to_idx_table()
    word_ids = word_to_idx_table.lookup(features[text_feature_name])

    embeddings = text_preprocessor.word_embeddings()

    new_features = {}
    new_features[text_feature_name] = tf.nn.embedding_lookup(
        embeddings, word_ids)

    labels = {k: tf.expand_dims(v, -1) for k, v in labels.items()}
    return old_model_fn(new_features, labels, mode=mode, config=config)

  old_config.replace(model_dir="/tmp/new_model")
  return tf.estimator.Estimator(
      new_model_fn, config=old_config, params=old_params)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--train_path",
      type=str,
      default="local_data/train.tfrecord",
      help="Path to the training data TFRecord file.")
  parser.add_argument(
      "--validate_path",
      type=str,
      default="local_data/validation.tfrecord",
      help="Path to the validation data TFRecord file.")
  parser.add_argument(
      "--embeddings_path",
      type=str,
      default="local_data/glove.6B/glove.6B.100d.txt",
      help="Path to the embeddings file.")

  args = parser.parse_args()
  train_path = types.Path(args.train_path)
  validate_path = types.Path(args.validate_path)
  embeddings_path = types.Path(args.embeddings_path)

  toxicity_q42017 = tfrecord_input.TFRecordInput(
      train_path=train_path,
      validate_path=validate_path,
      text_feature=FEATURE,
      labels=LABELS)
  preprocessor = text_preprocessor.TextPreprocessor(embeddings_path)

  keras_model = keras_rnn_model.KerasRNNModel(set(LABELS.keys())).get_model()

  # IMPORTANT: Model_to_estimator creates a checkpoint, however this checkpoint
  # does not contain the embedding variable (or other variables that we might
  # want to add outside of the Keras model). The workaround is to specify a
  # model_dir that is *not* the actual model_dir of the final model.
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=keras_model, model_dir="/tmp/keras_model")

  estimator = add_embedding_to_estimator(estimator, FEATURE, preprocessor)
  estimator.train(input_fn=toxicity_q42017.train_input_fn(), steps=10000)
  metrics = estimator.evaluate(
      input_fn=toxicity_q42017.validate_input_fn(), steps=100)
  print(metrics)


if __name__ == "__main__":
  main()
