# coding=utf-8
# Copyright 2018 The Conversation-AI.github.io Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
from tf_trainer.common.token_embedding_index import LoadTokenIdxEmbeddings
from typing import Callable, Dict, List, Optional, Tuple

FLAGS = flags.FLAGS

tf.app.flags.DEFINE_bool('is_embedding_trainable', False,
                         'Enable fine tuning of embeddings.')

class TextPreprocessor(object):
  """Text Preprocessor TensorFlow Estimator Extension.

  Uses embedding indexes to create tensors that map tokens (provided by an
  abstract tokenizer funtion) to embeddings.

  Note: Due to the lack of text preprocessing functions in tensorflow, we expect
  that the text is already preprocessed (list of words) in inference. In
  training, due to the availability of tf.py_func, we can handle the
  preprocessing.
  """

  def __init__(self, embeddings_path: str) -> None:
    self._word_to_idx, self._embeddings_matrix, self._unknown_token, self._embedding_size = \
      LoadTokenIdxEmbeddings(embeddings_path)  # type: Tuple[Dict[str, int], np.ndarray, int, int]

  def train_preprocess_fn(self,
                         tokenizer: Callable[[str], List[str]],
                         lowercase: Optional[bool] = True
                         ) -> Callable[[types.Tensor], types.Tensor]:

    def _tokenize(text: bytes) -> np.ndarray:
      """Converts text to a list of words.

      Args:
        text: text to tokenize (string).
        lowercase: whether to include lowercasing in preprocessing (boolean).
        tokenizer: Python function to tokenize the text on.

      Returns:
        A list of strings (words).
      """

      words = tokenizer(text.decode('utf-8'))
      if lowercase:
        words = [w.lower() for w in words]
      return np.asarray([
            self._word_to_idx.get(w, self._unknown_token)
            for w in words
        ])

    def _preprocess_fn(text: types.Tensor) -> types.Tensor:
      '''Converts a text into a list of integers.

      Args:
        text: a 0-D string Tensor.

      Returns:
        A 1-D int64 Tensor.
      '''
      words = tf.py_func(_tokenize, [text], tf.int64)
      return words

    return _preprocess_fn

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

    def add_init_fn_to_estimatorSpec(estimator_spec, init_fn):
      '''Add a new init_fn to the scaffold part of estimator spec.'''

      def new_init_fn(scaffold, sess):
        init_fn(scaffold, sess)
        if estimator_spec.scaffold.init_fn:
          estimator_spec.scaffold.init_fn(scaffold, sess)

      scaffold = tf.train.Scaffold(
          init_fn=new_init_fn,
          copy_from_scaffold=estimator_spec.scaffold)
      estimator_spec_with_scaffold = tf.estimator.EstimatorSpec(
          mode=estimator_spec.mode,
          predictions=estimator_spec.predictions,
          loss=estimator_spec.loss,
          train_op=estimator_spec.train_op,
          eval_metric_ops=estimator_spec.eval_metric_ops,
          export_outputs=estimator_spec.export_outputs,
          training_chief_hooks=estimator_spec.training_chief_hooks,
          training_hooks=estimator_spec.training_hooks,
          scaffold=scaffold,
          evaluation_hooks=estimator_spec.evaluation_hooks,
          prediction_hooks=estimator_spec.prediction_hooks
          )
      return estimator_spec_with_scaffold

    def new_model_fn(features, labels, mode, params, config):
      """model_fn used in defining the new TF Estimator"""

      embeddings, embedding_init_fn = self.word_embeddings(
          trainable=FLAGS.is_embedding_trainable)

      text_feature = features[text_feature_name]
      word_embeddings = tf.nn.embedding_lookup(embeddings, text_feature)
      new_features = {text_feature_name: word_embeddings}

      # Fix dimensions to make Keras model output match label dims.
      if mode != tf.estimator.ModeKeys.PREDICT:
        labels = {k: tf.expand_dims(v, -1) for k, v in labels.items()}

      # TODO: Modify when embeddings are part of the model.
      estimator_spec = old_model_fn(new_features, labels, mode=mode, config=config)
      estimator_spec_with_scaffold = add_init_fn_to_estimatorSpec(
          estimator_spec,
          embedding_init_fn)

      return estimator_spec_with_scaffold

    return tf.estimator.Estimator(
        new_model_fn, config=old_config, params=old_params)

  def word_to_idx(self) -> Dict[str, int]:
    return self._word_to_idx

  def unknown_token(self) -> int:
    return self._unknown_token

  def word_embeddings(self, trainable) -> tf.Variable:
    """Get word embedding TF Variable."""

    embeddings = tf.get_variable(
        "embeddings",
        self._embeddings_matrix.shape,
        trainable=trainable)

    def init_fn(scaffold, sess):
        sess.run(embeddings.initializer, {embeddings.initial_value: self._embeddings_matrix})

    return embeddings, init_fn
