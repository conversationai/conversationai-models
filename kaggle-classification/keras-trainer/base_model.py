"""Base model class used by the ModelRunner"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class BaseModel():
  """Base class for model runner"""

  def __init__(self):
    raise NotImplementedError('BaseModel should not be initialized.')

  def get_model(self):
    raise NotImplementedError('Method get_model needs to be implemented.')
