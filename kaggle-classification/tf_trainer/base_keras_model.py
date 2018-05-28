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

  def get_estimator(self):
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

    return estimator

  @staticmethod
  def roc_auc(y_true: types.Tensor, y_pred: types.Tensor,
              threshold=0.5) -> types.Tensor:
    """ ROC AUC based on TF's metrics package.
  
    We assume true labels are 'soft' and pick 0 or 1 based on a threshold.
    """
    y_true = tf.to_int32(tf.greater(y_true, threshold))
    value, update_op = tf.metrics.auc(y_true, y_pred)
    return update_op
