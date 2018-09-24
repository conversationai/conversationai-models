"""Experiments with Toxicity Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import common flags and run code. Must be imported first.
from tf_trainer.common import model_trainer

from tf_trainer.common import tfrecord_input
from tf_trainer.common import serving_input
from tf_trainer.common import text_preprocessor
from tf_trainer.common import types
from tf_trainer.tf_word_label_embedding import model as tf_word_label_embedding

import nltk
import tensorflow as tf

from typing import Dict

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("embeddings_path",
                           "local_data/glove.6B/glove.6B.100d.txt",
                           "Path to the embeddings file.")
tf.app.flags.DEFINE_boolean("is_binary_embedding", False,
                            "Whether embeddings are binaries.")
tf.app.flags.DEFINE_string("text_feature_name", "comment_text",
                           "Feature name of the text feature.")
tf.app.flags.DEFINE_string("key_name", "comment_key",
                           "Name of the key feature for serving examples.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "The batch size to use during training.")
tf.app.flags.DEFINE_integer("train_steps", 30000,
                            "The number of steps to train for.")
tf.app.flags.DEFINE_integer("eval_period", 500,
                            "The number of steps per eval period.")
tf.app.flags.DEFINE_integer("eval_steps", 50,
                            "The number of steps to eval for.")

# TODO: Missing fields are not handled properly yet.
LABELS = {
    "frac_neg": tf.float32,
    #"frac_very_neg": tf.float32
}  # type: Dict[str, tf.DType]


def main(argv):
  del argv  # unused

  embeddings_path = FLAGS.embeddings_path
  is_binary_embedding = FLAGS.is_binary_embedding
  text_feature_name = FLAGS.text_feature_name
  key_name = FLAGS.key_name

  embeddings_path = FLAGS.embeddings_path
  is_binary_embedding = FLAGS.is_binary_embedding
  text_feature_name = FLAGS.text_feature_name
  key_name = FLAGS.key_name

  preprocessor = text_preprocessor.TextPreprocessor(embeddings_path,
                                                    is_binary_embedding)

  nltk.download("punkt")
  train_preprocess_fn = preprocessor.train_preprocess_fn(nltk.word_tokenize)
  dataset = tfrecord_input.TFRecordInput(
      train_path=FLAGS.train_path,
      validate_path=FLAGS.validate_path,
      text_feature=text_feature_name,
      labels=LABELS,
      train_preprocess_fn=train_preprocess_fn,
      batch_size=FLAGS.batch_size,
      max_seq_len=5000)

  model_tf = tf_word_label_embedding.TFWordLabelEmbeddingModel(
      text_feature_name, "frac_neg")
  model = preprocessor.add_embedding_to_model(model_tf, text_feature_name)

  trainer = model_trainer.ModelTrainer(dataset, model)
  trainer.train_with_eval(FLAGS.train_steps, FLAGS.eval_period,
                          FLAGS.eval_steps)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
