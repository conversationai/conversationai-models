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
"""Interface for Models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from tf_trainer.common import types
from typing import Callable

# The TF Example key associated with input features that consist of an
# UTF-8 string, for models that use that as input.
TEXT_FEATURE_KEY = 'text'

# The TF Example key associated with a Tensor of int32s for models that
# use tokens from a vocabulary as input.
TOKENS_FEATURE_KEY = 'tokens'

# The TF Example key associated with examples in inference that consist of
# an int64 integer. It is a unique identifier of the TF Example and is passed
# along by the estimator and returned in the predictions (forward_features).
EXAMPLE_KEY = 'comment_key'


class BaseModel(abc.ABC):
  """Tentative interface for all model classes.

  Although the code doesn't take advantage of this interface yet, all models
  should subclass this one.
  """

  def map(self, f: Callable[[tf.estimator.Estimator], tf.estimator.Estimator]
         ) -> 'BaseModel':
    """Allows models to be extended. e.g.

    adding preprocessing steps.
    """

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
