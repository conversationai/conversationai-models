"""Interface for Models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from keras import models
from tf_trainer.common import text_preprocessor
from tf_trainer.common import types
from typing import Callable


class BaseModel(abc.ABC):
  """Tentative interface for all model classes.

  Although the code doesn't take advantage of this interface yet, all models
  should subclass this one.
  """

  def map(self, f: Callable[[tf.estimator.Estimator], tf.estimator.Estimator]
         ) -> 'BaseModel':

    class Model(BaseModel):

      def estimator(unused, model_dir):
        del unused
        return f(self.estimator(model_dir))

      def hparams(unused):
        del unused
        return self.hparams()

    return Model()

  @abc.abstractmethod
  def estimator(self, model_dir: str) -> tf.estimator.Estimator:
    pass

  def hparams(self) -> tf.contrib.training.HParams:
    return tf.contrib.training.HParams()
