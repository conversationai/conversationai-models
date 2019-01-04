"""Experiments with Toxicity Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
import tensorflow as tf

from tf_trainer.common import base_model
from tf_trainer.common import model_trainer
from tf_trainer.common import serving_input
from tf_trainer.common import text_preprocessor
from tf_trainer.common import tfrecord_input
from tf_trainer.common import types
from tf_trainer.tf_word_label_embedding import model as tf_word_label_embedding

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('embeddings_path',
                           'local_data/glove.6B/glove.6B.100d.txt',
                           'Path to the embeddings file.')


def main(argv):
  del argv  # unused

  preprocessor = text_preprocessor.TextPreprocessor(FLAGS.embeddings_path)

  nltk.download('punkt')
  train_preprocess_fn = preprocessor.train_preprocess_fn(nltk.word_tokenize)
  dataset = tfrecord_input.TFRecordInputWithTokenizer(
      train_preprocess_fn=train_preprocess_fn, max_seq_len=5000)

  model_tf = tf_word_label_embedding.TFWordLabelEmbeddingModel(dataset.labels())
  model = preprocessor.add_embedding_to_model(model_tf,
                                              base_model.TOKENS_FEATURE_KEY)

  trainer = model_trainer.ModelTrainer(dataset, model)
  trainer.train_with_eval()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
