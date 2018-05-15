"""Base model class used by the ModelRunner"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
from keras.layers import Input
from keras.models import Model


class BaseModel(metaclass=ABCMeta):
  """Base class for model runner"""

  @abstractmethod
  def get_model(self) -> Model:
    raise NotImplementedError('Method get_model needs to be implemented.')
