# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Defines the dataset structure for evaluation pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import random

import pandas as pd


class Model(object):

  def __init__(self,
               model_name,
               feature_keys,
               prediction_keys,
               example_key='example_key',
               model_version=None):
      self._model_name = model_name
      self._feature_keys= feature_keys
      self._prediction_keys = prediction_keys
      self._example_key = example_key
      self._model_version = model_version

  def feature_keys(self):
    return self._feature_keys


class Dataset(object):
  """Defines a format for every dataset to work with evaluation pipeline."""

  _SEED = 2018
  random.seed(_SEED)

  def __init__(self, input_fn, model):
    self.check_compatibility(input_fn, model)
    self._input_fn = input_fn
    self._model = model

  def check_compatibility(self, input_fn, model):
    """Asserts if the loading function of child class follows requirements."""
    loaded_data = input_fn(max_n_examples=1)

    if not isinstance(loaded_data, pd.DataFrame):
      raise ValueError('input_fn should return a pandas DataFrame.')

    if len(loaded_data) != 1:
      raise ValueError(
          'input_fn(max_n_examples=1) should contain 1 row (exactly).')
    
    for key in model.feature_keys():
      if key not in loaded_data.columns:
        raise ValueError(
            'input_fn must contain at least the feature keys {}'.format(
                model.feature_keys()))

