"""Text Preprocessor.

Used to tokenize and then use embeddings for text.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import nltk
import numpy as np
import tensorflow as tf
from tf_trainer import types
from typing import Tuple, Dict, Optional, List

FLAGS = flags.FLAGS


class TextPreprocessor():
  """Text Preprocessor.

  Takes an embedding and uses it to produce a preprocess method that can be
  called on texts to get its int representation. Also produces an embedding
  matrix to be used with a tensorflow graph.
  """

  UNKNOWN = '<UNK>'

  def __init__(self, embeddings_path: types.Path) -> None:
    nltk.download('punkt')
    self._word2idx, self._embeddings_matrix = TextPreprocessor._get_word_idx_and_embeddings(
        embeddings_path)  # type: Tuple[Dict[str, int], np.ndarray]

  def preprocess_text(self, text: str) -> List[int]:
    tokens = nltk.word_tokenize(text)
    return [
        self._word2idx.get(t, self._word2idx.get(TextPreprocessor.UNKNOWN))
        for t in tokens
    ]

  def embeddings_matrix(self, trainable: bool) -> tf.Variable:
    """Returns embedding tf variable.

    Use tf.nn.embedding_lookup to get embeddings.

    Args:
      word_embeddings: Numpy array of word embeddings.

    Returns:
      TF Variable initialized with the embeddings.
    """

    embeddings_shape = self._embeddings_matrix.shape,
    initial_embeddings_matrix = tf.constant_initializer(self._embeddings_matrix)
    return tf.get_variable(
        name='word_embeddings',
        shape=embeddings_shape,
        initializer=initial_embeddings_matrix,
        trainable=trainable)

  @staticmethod
  def _get_word_idx_and_embeddings(embeddings_path: types.Path,
                                   max_words: Optional[int] = None
                                  ) -> Tuple[Dict[str, int], np.ndarray]:
    """Generate word to idx mapping and word embeddings numpy array.

    Args:
      embeddings_path: Local, GCS, or HDFS path to embedding file. Each line
        should be a word and its vector representation separated by a space.
      max_words: The max number of words we are going to allow as part of the
        embedding.

    Returns:
      Tuple of ord to idx mapping and Numpy array of word embeddings with shape
      (vocab size, embedding size).
    """
    word2idx = {}
    word_embeddings = []
    with tf.gfile.Open(embeddings_path, 'r') as f:
      for idx, line in f:
        if max_words and idx >= max_words:
          break

        values = line.split()
        word = values[0]
        word_embedding = np.asarray(values[1:], dtype='float32')
        word2idx[word] = idx
        word_embeddings.append(word_embedding)

    word2idx[TextPreprocessor.UNKNOWN] = len(word_embeddings)
    embeddings_matrix = np.asarray(word_embeddings, dtype=np.float32)
    np.append(embeddings_matrix, [embeddings_matrix.mean(axis=0)], axis=0)
    return word2idx, embeddings_matrix
