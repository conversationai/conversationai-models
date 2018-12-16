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
"""Tests for text_preprocessor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_trainer.common import text_preprocessor

class TextPreprocessorTest(tf.test.TestCase):

  def test_Tokenize(self):
    preprocessor = text_preprocessor.TextPreprocessor(
        'testdata/cats_and_dogs_onehot.vocab.txt')
    with self.test_session() as session:
      preprocess_fn = preprocessor.train_preprocess_fn(
          tokenizer=lambda x: x.split(' '), lowercase=False)
      tokens = preprocess_fn('dogs good cats bad rabbits not')
      self.assertEqual(list(tokens.eval()), [1, 3, 2, 4, 7, 6])

  def test_Lowercase(self):
    preprocessor = text_preprocessor.TextPreprocessor(
        'testdata/cats_and_dogs_onehot.vocab.txt')
    with self.test_session() as session:
      preprocess_fn = preprocessor.train_preprocess_fn(
          tokenizer=lambda x: x.split(' '), lowercase=True)
      tokens = preprocess_fn('Dogs GOOD Cats BAD rabbits not')
      self.assertEqual(list(tokens.eval()), [1, 3, 2, 4, 7, 6])


if __name__ == '__main__':
  tf.test.main()
