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

  def __init__(self, 
               embeddings_path: str, 
               is_binary_embedding: Optional[bool] = False
               ) -> None:
    self._word_to_idx, self._embeddings_matrix, self._unknown_token, self._embedding_size = \
        TextPreprocessor._get_word_idx_and_embeddings(
            embeddings_path,
            is_binary_embedding)  # type: Tuple[Dict[str, int], np.ndarray, int]

  def tokenize_tensor_op_tf_func(self) -> Callable[[types.Tensor], types.Tensor]:
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
        text: must be a 1-D Tensor string tensor.

      Returns:
        A 2-D Tensor of word integers.
      '''

      # TODO: Ensure utf-8 encoding. Currently the string is parsed with default encoding (unclear).
      text = tf.regex_replace(
          text,
          '[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]',
          ' ')
      words = tf.string_split(text)
      words_int_sparse = vocabulary_table.lookup(words)
      words_int_dense = tf.sparse_tensor_to_dense(
          words_int_sparse,
          default_value=0)
      return words_int_dense

    return _tokenize_tensor_op

  def tokenize_tensor_op_py_func(self, 
                                tokenizer: Callable[[str], List[str]],
                                lowercase: Optional[bool] = True
                                ) -> Callable[[types.Tensor], types.Tensor]:
    """Tensor op that converts some text into an array of ints that correspond
    with this preprocessor's embedding.

    This function is implemented with python operations and is therefore
    incompatible with TF-Serving (vs. tokenize_tensor_op_tf_func_).
    """

    def _tokenize_tensor_op(text: types.Tensor) -> types.Tensor:
      '''Converts a string Tensor to an array of integers.

      Args:
        text: must be a 1-D Tensor string tensor.

      Returns:
        A 2-D Tensor of word integers.
      '''

      def _tokenize(sentences: List[bytes]) -> np.ndarray:
        
        if len(sentences) > 1:
          raise ValueError('py_tokenizer does not support padding for now'
                           ' and can not be run on multiple sentences.')

        sentence = sentences[0]
        words = tokenizer(sentence.decode('utf-8'))
        if lowercase:
          words = [w.lower() for w in words]
        return np.asarray([[
            self._word_to_idx.get(w, self._unknown_token)
            for w in words
        ]])

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

    Note: We need to consider the case of large embeddings (see: 
      https://stackoverflow.com/questions/48217599/
      how-to-initialize-embeddings-layer-within-estimator-api/48243086#48243086).

    """
    old_model_fn = estimator.model_fn
    old_config = estimator.config
    old_params = estimator.params

    def add_init_fn_and_hook_to_estimatorSpec(estimator_spec, init_fn, training_hook):
      '''Add a new init_fn to the scaffold part of estimator spec.'''

      def new_init_fn(scaffold, sess):
        init_fn(scaffold, sess)
        if estimator_spec.scaffold.init_fn:
          estimator_spec.scaffold.init_fn(scaffold, sess)
      
      scaffold = tf.train.Scaffold(
        init_fn=new_init_fn,
        copy_from_scaffold=estimator_spec.scaffold)
      new_hook = list(estimator_spec.training_hooks).append(training_hook)
      estimator_spec_with_scaffold = tf.estimator.EstimatorSpec(
          mode=estimator_spec.mode,
          predictions=estimator_spec.predictions,
          loss=estimator_spec.loss,
          train_op=estimator_spec.train_op,
          eval_metric_ops=estimator_spec.eval_metric_ops,
          export_outputs=estimator_spec.export_outputs,
          training_chief_hooks=estimator_spec.training_chief_hooks,
          training_hooks=new_hook, #estimator_spec.training_hooks,
          scaffold=scaffold,
          evaluation_hooks=estimator_spec.evaluation_hooks,
          prediction_hooks=estimator_spec.prediction_hooks
          )
      return estimator_spec_with_scaffold

    def new_model_fn(features, labels, mode, params, config):
      """model_fn used in defining the new TF Estimator"""

      embeddings, embedding_init_fn = self.word_embeddings()

      text_feature = features[text_feature_name]
      unknown_fraction = tf.reduce_mean(
          tf.cast(
              tf.equal(text_feature, self._unknown_token),
              tf.float32)
          )
      tf.summary.scalar("fraction_of_unknown_words", unknown_fraction)  
      word_embeddings = tf.nn.embedding_lookup(embeddings, text_feature)
      new_features = {text_feature_name: word_embeddings}

      # Fix dimensions to make Keras model output match label dims.
      if mode != tf.estimator.ModeKeys.PREDICT:
        labels = {k: tf.expand_dims(v, -1) for k, v in labels.items()}

      summary_hook = tf.train.SummarySaverHook(
          save_steps=100,
          summary_op=tf.summary.merge_all())

      # TODO: Modify when embeddings are part of the model.
      estimator_spec = old_model_fn(new_features, labels, mode=mode, config=config)
      estimator_spec_with_scaffold = add_init_fn_and_hook_to_estimatorSpec(
          estimator_spec,
          embedding_init_fn,
          summary_hook)

      return estimator_spec_with_scaffold

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

    embeddings = tf.get_variable(
        "embeddings",
        self._embeddings_matrix.shape,
        trainable=trainable)
    
    def init_fn(scaffold, sess):
        sess.run(embeddings.initializer, {embeddings.initial_value: self._embeddings_matrix})
    
    return embeddings, init_fn

  @staticmethod
  def _get_word_idx_and_embeddings(embeddings_path: str,
                                   is_binary_embedding: bool,
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
    if is_binary_embedding:
      read_mode = 'rb'
    else:
      read_mode = 'r'
    with tf.gfile.Open(embeddings_path, read_mode) as f:
      for idx, line in enumerate(f):
        if max_words and idx >= max_words:
          break

        values = line.split()
        # Remove header when necessary.
        if len(values) == 2 and idx == 0:
          continue
        word = values[0]
        word_embedding = np.asarray(values[1:], dtype='float32')
        word_to_idx[word] = idx + 1  # Reserve first row for padding
        word_embeddings.append(word_embedding)

    # Add the padding "embedding"
    word_embeddings.insert(0, np.random.randn(len(word_embeddings[0])))

    # Convert embedding to numpy array and append the unknown word embedding,
    # which is the mean of all other embeddings.
    unknown_token = len(word_embeddings)
    try:
      embeddings_matrix = np.asarray(word_embeddings, dtype=np.float32)
    except:
      raise Exception('Embeddings can not be initialized.'
                      ' Is embedding binary = {}?'.format(is_binary_embedding))
    embeddings_matrix = np.append(
        embeddings_matrix, [embeddings_matrix.mean(axis=0)], axis=0)
    return word_to_idx, embeddings_matrix, unknown_token, len(word_embeddings[0])
