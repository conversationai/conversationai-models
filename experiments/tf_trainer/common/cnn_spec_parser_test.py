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
"""Tests for tfrecord_input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_trainer.common.cnn_spec_parser import SequentialLayers
from tf_trainer.common.cnn_spec_parser import ConcurrentFilters
from tf_trainer.common.cnn_spec_parser import Filter

class CnnSpecParserTest(tf.test.TestCase):

  def test_SequentialLayers(self):
    s = ('(2 / 2 -> 100), (3 / 2 -> 101) '
         ': (6 / 2 -> 102) '
         ': (3 / 1 -> 103)')
    spec = SequentialLayers(s)
    layer0 = spec.layers[0]
    self.assertEqual(len(layer0.filters), 2)
    layer0filter0 = layer0.filters[0] # type: Filter
    self.assertEqual(layer0filter0.size, 2)
    self.assertEqual(layer0filter0.stride, 2)
    self.assertEqual(layer0filter0.num_filters, 100)
    self.assertEqual(str(spec), s)

if __name__ == "__main__":
  tf.test.main()
