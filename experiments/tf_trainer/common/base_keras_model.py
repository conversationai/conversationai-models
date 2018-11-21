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
"""Abstract Base Class for Keras Models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from keras import models
from tf_trainer.common import types
from tf_trainer.common import base_model


class BaseKerasModel(base_model.BaseModel):
  """Abstract Base Class for Keras Models.

  Interface for Keras models.
  """
  TMP_MODEL_DIR = '/tmp/keras_model'

  @abc.abstractmethod
  def _get_keras_model(self) -> models.Model:
    """Compiled Keras model.

    Inputs should be word embeddings.
    """
    pass

  def estimator(self, model_dir):
    """Estimator created based on this instances Keras model.

    The generated estimator expected a tokenized text input (i.e. a sequence of
    words), and is responsible for generating the embedding with the provided
    preprocessor).
    """

    keras_model = self._get_keras_model()

    # IMPORTANT: model_to_estimator creates a checkpoint, however this checkpoint
    # does not contain the embedding variable (or other variables that we might
    # want to add outside of the Keras model). The workaround is to specify a
    # model_dir that is *not* the actual model_dir of the final model.
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=keras_model, model_dir=BaseKerasModel.TMP_MODEL_DIR)

    new_config = estimator.config.replace(model_dir=model_dir)

    # Why does estimator.model_fn not include params...
    def new_model_fn(features, labels, mode, params, config):
      return estimator.model_fn(features, labels, mode, config)

    return tf.estimator.Estimator(
        new_model_fn, config=new_config, params=estimator.params)

  @staticmethod
  def roc_auc(y_true: types.Tensor, y_pred: types.Tensor,
              threshold=0.5) -> types.Tensor:
    """ROC AUC based on TF's metrics package. This provides AUC in a Keras
    metrics compatible way (Keras doesn't have AUC otherwise).

    We assume true labels are 'soft' and pick 0 or 1 based on a threshold.
    """
    y_bool_true = tf.greater(y_true, threshold)
    value, update_op = tf.metrics.auc(y_bool_true, y_pred)
    return update_op
