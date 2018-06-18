"""Text Preprocessor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import nltk
import numpy as np
import functools
import tensorflow as tf
from tf_trainer.common import types
from typing import Tuple, Dict, Optional, List, Callable

FLAGS = flags.FLAGS


class TextPreprocessor():
  """Text Preprocessor.

  Takes an embedding and uses it to produce a word to index mapping and an
  embedding matrix.

  NOTE: You might be wondering why we don't go straight from the word to the
  embedding. The (maybe incorrect) thought process is that
  the embedding portion can be made a part of the tensorflow graph whereas the
  word to index portion can not (since words have variable length). Future work
  may include fixing a max word length.
  """

  def __init__(self, embeddings_path: str) -> None:
    self._word_to_idx, self._embeddings_matrix, self._unknown_token = TextPreprocessor._get_word_idx_and_embeddings(
        embeddings_path)  # type: Tuple[Dict[str, int], np.ndarray, int]

  def tokenize_tensor_op(self, tokenizer: Callable[[str], List[str]]
                        ) -> Callable[[types.Tensor], types.Tensor]:
    """Tensor op that converts some text into an array of ints that correspond
    with this preprocessor's embedding.
    """

    def _tokenize_tensor_op(text: types.Tensor) -> types.Tensor:

      def _tokenize(b: bytes) -> np.ndarray:
        return np.asarray([
            self._word_to_idx.get(w, self._unknown_token)
            for w in tokenizer(b.decode('utf-8'))
        ])

      return tf.py_func(_tokenize, [text], tf.int64)

    return _tokenize_tensor_op

  def create_estimator_with_embedding(
      self,
      estimator: tf.estimator.Estimator,
      text_feature_name: str,
      model_dir: str = '/tmp/new_model') -> tf.estimator.Estimator:
    """Takes an existing estimator and prepends the embedding layers to it.

    Args:
      estimator: A predefined Estimator that expects embeddings.
      text_feature_name: The name of the feature containing the text.
      model_dir: Place to output estimator model files.

    Returns:
      TF Estimator with embedding ops added.
    """
    old_model_fn = estimator.model_fn
    old_config = estimator.config
    old_params = estimator.params

    def new_model_fn(features, labels, mode, params, config):
      """model_fn used in defining the new TF Estimator"""

      embeddings = self.word_embeddings()

      text_feature = features[text_feature_name]
      text_feature = tf.pad(text_feature, [[0, 0], [0, 300]])
      text_feature = text_feature[:, 0:300]
      word_embeddings = tf.nn.embedding_lookup(embeddings, text_feature)
      new_features = {text_feature_name: word_embeddings}

      # Fix dimensions to make Keras model output match label dims.
      labels = {k: tf.expand_dims(v, -1) for k, v in labels.items()}
      return old_model_fn(
          new_features, labels['frac_neg'], mode=mode, config=config)

    return tf.estimator.Estimator(
        new_model_fn, config=old_config, params=old_params)

  def word_to_idx(self) -> Dict[str, int]:
    return self._word_to_idx

  def unknown_token(self) -> int:
    return self._unknown_token

  def word_to_idx_table(self) -> tf.contrib.lookup.HashTable:
    """Get word to index mapping as a TF HashTable."""

    keys = list(self._word_to_idx.keys())
    values = list(self._word_to_idx.values())
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
        self._unknown_token)
    return table

  def word_embeddings(self, trainable=True) -> tf.Variable:
    """Get word embedding TF Variable."""

    embeddings_shape = self._embeddings_matrix.shape
    initial_embeddings_matrix = tf.constant_initializer(self._embeddings_matrix)
    embeddings = tf.get_variable(
        name='word_embeddings',
        shape=embeddings_shape,
        initializer=initial_embeddings_matrix,
        trainable=trainable)
    return embeddings

  @staticmethod
  def _get_word_idx_and_embeddings(embeddings_path: str,
                                   max_words: Optional[int] = None
                                  ) -> Tuple[Dict[str, int], np.ndarray, int]:
    """Generate word to idx mapping and word embeddings numpy array.

    We have two levels of indirection (e.g. word to idx and then idx to
    embedding) which could reduce embedding size if multiple words map to the
    same idx. This is not currently a use case.

    Args:
      embeddings_path: Local, GCS, or HDFS path to embedding file. Each line
        should be a word and its vector representation separated by a space.
      max_words: The max number of words we are going to allow as part of the
        embedding.

    Returns:
      Tuple of vocab list, Numpy array of word embeddings with shape
      (vocab size, embedding size), and the unknown token.
    """
    word_to_idx = {}
    word_embeddings = []
    with tf.gfile.Open(embeddings_path) as f:
      for idx, line in enumerate(f):
        if max_words and idx >= max_words:
          break

        values = line.split()
        word = values[0]
        word_embedding = np.asarray(values[1:], dtype='float32')
        word_to_idx[word] = idx + 1  # Reserve first row for padding
        word_embeddings.append(word_embedding)

    # Add the padding "embedding"
    word_embeddings.insert(0, np.random.randn(len(word_embeddings[0])))

    # Convert embedding to numpy array and append the unknown word embedding,
    # which is the mean of all other embeddings.
    unknown_token = len(word_embeddings)
    embeddings_matrix = np.asarray(word_embeddings, dtype=np.float32)
    embeddings_matrix = np.append(
        embeddings_matrix, [embeddings_matrix.mean(axis=0)], axis=0)
    return word_to_idx, embeddings_matrix, unknown_token
