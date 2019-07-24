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

from tf_trainer.common.token_embedding_index import LoadTokenIdxEmbeddings


class LoadTokenIdxEmbeddingsTest(tf.test.TestCase):

  def test_LoadTokenIdxEmbeddings(self):
    idx, embeddings, unknown_idx, embedding_size = LoadTokenIdxEmbeddings(
        'testdata/cats_and_dogs_onehot.vocab.txt')
    self.assertEqual(embedding_size, 6)
    self.assertEqual(unknown_idx, 7)
    self.assertEqual(idx['dogs'], 1)
    self.assertEqual(idx['cats'], 2)
    self.assertEqual(idx['not'], 6)
    self.assertEqual(embeddings[1][0], 1.0)
    self.assertEqual(embeddings[1][1], 0.0)
    # Note: padding embedding will be random, and is index 0. Also the unknown
    # token embedding will be random, and is index n+1; 7 in this case.

  def test_LoadTokenIdxEmbeddingsGlove300d(self):
    idx, embeddings, unknown_idx, embedding_size = LoadTokenIdxEmbeddings(
        'gs://kaggle-model-experiments/resources/glove.6B/glove.6B.300d.txt')
    self.assertEqual(embedding_size, 300)


if __name__ == '__main__':
  tf.test.main()
