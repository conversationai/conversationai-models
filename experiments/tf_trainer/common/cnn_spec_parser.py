"""CNN Specification Parser

Parsers the specification of convolutional layers (the number and sizes of
filters)

```
  layers = layer : layers
  layer = filters
  filters = filter, filters
  filter = numfilters * (size / stride -> output_embedding_size)
  size, stride, output_embedding_size = \d+
```
"""

import re
from typing import List

filter_regexp = re.compile(
  r'(?P<num_filters>\d+)\s*\*\s*'
  r'\(\s*(?P<size>\d+)\s*/\s*(?P<stride>\d+)\s*'
  r'\-\>\s*(?P<output_embedding_size>\d+)\s*\)')

layers_split_regexp = re.compile(r'\s*:\s*')

filters_split_regexp = re.compile(r'\s*,\s*')

class FilterParseError(Exception):
  pass

class Filter(object):
  """A single CNN filter.

  filter = 'num_filters * (size / stride -> output_embedding_size)'
  """
  # def parse(self, str:str):

  def __init__(self, str:str) -> None:
    m = filter_regexp.match(str)
    if m is None:
      raise FilterParseError(f'Bad filter definition for: {str}')
    self.num_filters = int(m.group('num_filters')) # type "int"
    self.size = int(m.group('size')) # type "int"
    self.stride = int(m.group('stride')) # type "int"
    self.output_embedding_size = int(m['output_embedding_size']) # type "int"

  def __str__(self) -> str:
    return (f'{self.num_filters} * '
      f'({self.size} / {self.stride} -> {self.output_embedding_size})')

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
