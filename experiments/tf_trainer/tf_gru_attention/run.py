"""Experiments with Toxicity Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import common flags and run code. Must be imported first.
from tf_trainer.common import model_runner

from tf_trainer.common import tfrecord_input
from tf_trainer.common import text_preprocessor
from tf_trainer.common import types
from tf_trainer.tf_gru_attention import model as tf_gru_attention

import nltk
import tensorflow as tf

from typing import Dict

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("embeddings_path",
                           "local_data/glove.6B/glove.6B.100d.txt",
                           "Path to the embeddings file.")
tf.app.flags.DEFINE_string("text_feature_name", "comment_text",
                           "Feature name of the text feature.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "The batch size to use during training.")
tf.app.flags.DEFINE_integer("train_steps", 5000,
                            "The number of steps to train for.")
tf.app.flags.DEFINE_integer("eval_period", 200,
                            "The number of steps per eval period.")
tf.app.flags.DEFINE_integer("eval_steps", 100,
                            "The number of steps to eval for.")

# TODO: Missing fields are not handled properly yet.
LABELS = {
    "frac_neg": tf.float32,
    #"frac_very_neg": tf.float32
}  # type: Dict[str, tf.DType]


def main(argv):
  del argv  # unused

  embeddings_path = FLAGS.embeddings_path
  text_feature_name = FLAGS.text_feature_name

  preprocessor = text_preprocessor.TextPreprocessor(embeddings_path)
  nltk.download("punkt")
  tokenize_op = preprocessor.tokenize_tensor_op(nltk.word_tokenize)

  dataset = tfrecord_input.TFRecordInput(
      train_path=FLAGS.train_path,
      validate_path=FLAGS.validate_path,
      text_feature=text_feature_name,
      labels=LABELS,
      feature_preprocessor=tokenize_op,
      batch_size=FLAGS.batch_size)

  model = preprocessor.add_embedding_to_model(
      tf_gru_attention.TFRNNModel(text_feature_name, LABELS), text_feature_name)

  runner = model_runner.ModelRunner(
      dataset, model,
      tf_gru_attention.TFRNNModel.hparams().values())
  runner.train_with_eval(FLAGS.train_steps, FLAGS.eval_period, FLAGS.eval_steps)


if __name__ == "__main__":
  tf.app.run(main)
