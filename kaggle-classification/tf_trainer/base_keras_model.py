"""Abstract Base Class for DatasetInput."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from keras import models
from tf_trainer import text_preprocessor
from tf_trainer import types


class BaseKerasModel(abc.ABC):
  """Abstract Base Class for Keras Models.

  Interface for Keras models.
  """

  @abc.abstractmethod
  def _get_keras_model(self) -> models.Model:
    """Compiled Keras model.

    Inputs should be word embeddings.
    """
    pass

  def get_estimator(self, preprocessor: text_preprocessor.TextPreprocessor,
                    text_feature_name: str):
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
        keras_model=keras_model, model_dir="/tmp/keras_model")

    estimator = BaseKerasModel._create_estimator_with_embedding(
        estimator, text_feature_name, preprocessor)

    return estimator

  @staticmethod
  def _create_estimator_with_embedding(
      estimator: tf.estimator.Estimator,
      text_feature_name: str,
      text_preprocessor: text_preprocessor.TextPreprocessor,
      model_dir: str = "/tmp/new_model") -> tf.estimator.Estimator:
    """Takes an existing estimator and prepends the embedding layers to it.
  
    Args:
      estimator: A predefined Estimator that expects embeddings.
      text_feature_name: The name of the feature containing the text.
      text_preprocess: An instance of TextPreprocessor holding embedding info.
  
    Returns:
      TF Estimator with embedding ops added.
    """
    old_model_fn = estimator.model_fn
    old_config = estimator.config
    old_params = estimator.params

    def new_model_fn(features, labels, mode, params, config):
      """model_fn used in defining the new TF Estimator"""

      word_to_idx_table = text_preprocessor.word_to_idx_table()
      word_ids = word_to_idx_table.lookup(features[text_feature_name])

      embeddings = text_preprocessor.word_embeddings()

      new_features = {}
      new_features[text_feature_name] = tf.nn.embedding_lookup(
          embeddings, word_ids)

      # Fix dimensions to make Keras model output match label dims.
      labels = {k: tf.expand_dims(v, -1) for k, v in labels.items()}
      return old_model_fn(new_features, labels, mode=mode, config=config)

    old_config = old_config.replace(model_dir=model_dir)
    return tf.estimator.Estimator(
        new_model_fn, config=old_config, params=old_params)

  @staticmethod
  def roc_auc(y_true: types.Tensor, y_pred: types.Tensor,
              threshold=0.5) -> types.Tensor:
    """ ROC AUC based on TF's metrics package.
  
    We assume true labels are 'soft' and pick 0 or 1 based on a threshold.
    """

    y_true = tf.to_int32(tf.greater(y_true, threshold))
    value, update_op = tf.metrics.auc(y_true, y_pred)
    return update_op
