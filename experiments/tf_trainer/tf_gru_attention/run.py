"""Experiments with Toxicity Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_trainer.common import tfrecord_input
from tf_trainer.common import text_preprocessor
from tf_trainer.common import types
from tf_trainer.common import model_runner
from tf_trainer.tf_gru_attention import model

import nltk
import tensorflow as tf

from typing import Dict

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("embeddings_path",
                           "local_data/glove.6B/glove.6B.100d.txt",
                           "Path to the embeddings file.")
tf.app.flags.DEFINE_string("text_feature_name", "comment_text",
                           "Feature name of the text feature.")

# TODO: Missing fields are not handled properly yet.
LABELS = {
    "frac_neg": tf.float32,
    #"frac_very_neg": tf.float32
}  # type: Dict[str, tf.DType]


class TFGRUAttentionModelRunner(model_runner.ModelRunner):

  def __init__(self, embeddings_path: str, text_feature: str,
               labels: Dict[str, tf.DType],
               text_preprocessor: text_preprocessor.TextPreprocessor) -> None:
    self._embeddings_path = embeddings_path
    self._text_feature = text_feature
    self._labels = labels
    self._text_preprocessor = text_preprocessor
    nltk.download("punkt")

  def dataset_input(self, train_path, validate_path):
    return tfrecord_input.TFRecordInput(
        train_path=train_path,
        validate_path=validate_path,
        text_feature=self._text_feature,
        labels=self._labels,
        feature_preprocessor=self._text_preprocessor.tokenize_tensor_op(
            nltk.word_tokenize))

  def estimator(self, model_dir):
    estimator_no_embedding = model.TFRNNModel(
        self._text_feature, self._labels).estimator(
            tf.estimator.RunConfig(model_dir=model_dir))

    estimator = self._text_preprocessor.create_estimator_with_embedding(
        estimator_no_embedding, self._text_feature)

    return estimator


def main(argv):
  del argv  # unused

  embeddings_path = FLAGS.embeddings_path
  text_feature_name = FLAGS.text_feature_name

  preprocessor = text_preprocessor.TextPreprocessor(embeddings_path)

  runner = TFGRUAttentionModelRunner(
      embeddings_path=embeddings_path,
      text_feature=text_feature_name,
      labels=LABELS,
      text_preprocessor=preprocessor)

  runner.train_with_eval(20000, 1000, 100)


if __name__ == "__main__":
  tf.app.run(main)
