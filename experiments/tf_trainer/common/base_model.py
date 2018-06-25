"""Interface for Models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from keras import models
from tf_trainer.common import text_preprocessor
from tf_trainer.common import types


class BaseModel(abc.ABC):
  """Tentative interface for all model classes.

  Although the code doesn't take advantage of this interface yet, all models
  should subclass this one.
  """

  @abc.abstractmethod
  def estimator(self, model_dir) -> tf.estimator.Estimator:
    pass
