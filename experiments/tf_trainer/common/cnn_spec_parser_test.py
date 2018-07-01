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
    s = ('20 * (2 / 2 -> 100), 10 * (3 / 2 -> 100) '
         ': 20 * (6 / 2 -> 100) '
         ': 10 * (3 / 1 -> 100)')
    spec = SequentialLayers(s)
    layer0 = spec.layers[0]
    self.assertEqual(len(layer0.filters), 2)
    layer0filter0 = layer0.filters[0] # type: Filter
    self.assertEqual(layer0filter0.size, 2)
    self.assertEqual(layer0filter0.stride, 2)
    self.assertEqual(layer0filter0.output_embedding_size, 100)
    self.assertEqual(layer0filter0.num_filters, 20)
    self.assertEqual(str(spec), s)

if __name__ == "__main__":
  tf.test.main()
