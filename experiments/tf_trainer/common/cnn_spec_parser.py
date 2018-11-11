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
"""CNN Specification Parser.

A simple parser for specifications of convolutional layers.

BNF defining the syntax to specify CNNs:
```
  layers = layer : layers
  layer = filters
  filters = filter, filters
  filter = (size / stride -> num_filters)
  size, stride, num_filters = \d+
```

Note that num_filters is the output embedding size.
"""

import re
from typing import List


layers_split_regexp = re.compile(r'\s*:\s*')
filters_split_regexp = re.compile(r'\s*,\s*')
filter_regexp = re.compile(
  r'\(\s*(?P<size>\d+)\s*/\s*(?P<stride>\d+)\s*'
  r'\-\>\s*(?P<num_filters>\d+)\s*\)')


class FilterParseError(Exception):
  pass


class Filter(object):
  """A single CNN filter.

  filter = '(size / stride -> num_filters)'
  """

  def __init__(self, str:str) -> None:
    m = filter_regexp.match(str)
    if m is None:
      raise FilterParseError(f'Bad filter definition for: {str}')
    self.num_filters = int(m.group('num_filters')) # type "int"
    self.size = int(m.group('size')) # type "int"
    self.stride = int(m.group('stride')) # type "int"

  def __str__(self) -> str:
    return (
      f'({self.size} / {self.stride} -> {self.num_filters})')


class ConcurrentFilters(object):
  """A set of concurrent CNN filters that make up one layer

  filters = filter, filters
  """
  def __init__(self, str:str) -> None:
    filter_spec_strs = filters_split_regexp.split(str)
    self.filters = [Filter(s) for s in filter_spec_strs]

  def __str__(self) -> str:
    return ', '.join([str(f) for f in self.filters])


class SequentialLayers(object):
  """A sequence of CNN layers
  layers = filters : layers
  """
  def __init__(self, str:str) -> None:
    layer_spec_strs = layers_split_regexp.split(str)
    self.layers = [ConcurrentFilters(s) for s in layer_spec_strs]  # type: List[ConcurrentFilters]

  def __str__(self) -> str:
    return ' : '.join([str(f) for f in self.layers])
