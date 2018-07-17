"""Text Preprocessor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import flags
import numpy as np
import tensorflow as tf
from tf_trainer.common import base_model
from tf_trainer.common import types
from typing import Callable, Dict, List, Optional, Tuple

FLAGS = flags.FLAGS


class TextPreprocessor(object):
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
    self._word_to_idx, self._embeddings_matrix, self._unknown_token = \
        TextPreprocessor._get_word_idx_and_embeddings(
            embeddings_path)  # type: Tuple[Dict[str, int], np.ndarray, int]

  def tokenize_tensor_op_tf_func(self, single_record_level=True) -> Callable[[types.Tensor], types.Tensor]:
    """Tensor op that converts some text into an array of ints that correspond
    with this preprocessor's embedding.

    This function is implemented only with TensorFlow operations and is
    therefore compatible with TF-Serving (vs. tokenize_tensor_op_py_func).
    """

    vocabulary_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            keys=list(self._word_to_idx.keys()),
            values=list(self._word_to_idx.values()),
            key_dtype=tf.string,
            value_dtype=tf.int64),
        default_value=self._unknown_token)

    def _tokenize_tensor_op(text: types.Tensor) -> types.Tensor:
      '''Converts a string Tensor to an array of integers.

      Args:
        text: must be a scalar string tensor (rank 0).

      Returns:
        A 1-D Tensor of word integers.
      '''

      # TODO: Improve tokenizer.
      # TODO: Ensure utf-8 encoding. Currently the string is parsed with default encoding (unclear). 
      if single_record_level:
        words = tf.string_split([text])
      else:
        words = tf.string_split(text)
      words_int_sparse = vocabulary_table.lookup(words)
      words_int_dense = tf.sparse_to_dense(
          words_int_sparse.indices,
          words_int_sparse.dense_shape,
          words_int_sparse.values,
          default_value=0)

      if single_record_level:
        return tf.squeeze(words_int_dense)
      else:
        return words_int_dense

    return _tokenize_tensor_op

  def tokenize_tensor_op_py_func(self, tokenizer: Callable[[str], List[str]]
                                ) -> Callable[[types.Tensor], types.Tensor]:
    """Tensor op that converts some text into an array of ints that correspond
    with this preprocessor's embedding.

    This function is implemented with python operations and is therefore
    incompatible with TF-Serving (vs. tokenize_tensor_op_tf_func_).
    """

    def _tokenize_tensor_op(text: types.Tensor) -> types.Tensor:

      def _tokenize(b: bytes) -> np.ndarray:
        return np.asarray([
            self._word_to_idx.get(w, self._unknown_token)
            for w in tokenizer(b.decode('utf-8'))
        ])

      return tf.py_func(_tokenize, [text], tf.int64)

    return _tokenize_tensor_op

  def add_embedding_to_model(self, model: base_model.BaseModel,
                             text_feature_name: str) -> base_model.BaseModel:
    """Returns a new BaseModel with an embedding layer prepended.

    Args:
      model: An existing BaseModel instance.
      text_feature_name: The name of the feature containing text.
    """

    return model.map(
        functools.partial(self.create_estimator_with_embedding,
                          text_feature_name))

  def create_estimator_with_embedding(
      self, text_feature_name: str,
      estimator: tf.estimator.Estimator) -> tf.estimator.Estimator:
    """Takes an existing estimator and prepends the embedding layers to it.

    Args:
      estimator: A predefined Estimator that expects embeddings.
      text_feature_name: The name of the feature containing the text.

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
      # Make sure all examples are length 300
      # TODO: Parameterize 300
      text_feature = tf.pad(text_feature, [[0, 0], [0, 300]])
      text_feature = text_feature[:, 0:300]
      word_embeddings = tf.nn.embedding_lookup(embeddings, text_feature)
      new_features = {text_feature_name: word_embeddings}

      # Fix dimensions to make Keras model output match label dims.
      if mode != tf.estimator.ModeKeys.PREDICT:
        labels = {k: tf.expand_dims(v, -1) for k, v in labels.items()}

      return old_model_fn(new_features, labels, mode=mode, config=config)

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
