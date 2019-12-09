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
"""Working with Token Embeding Indexes."""

from typing import Tuple, Dict, Optional, List, Callable
import numpy as np
import functools
import tensorflow as tf

def LoadTokenIdxEmbeddings(embeddings_path: str) \
  -> Tuple[Dict[str, int], np.ndarray, int, int]:
  """Generate word to idx mapping and word embeddings numpy array.

  We have two levels of indirection (e.g. word to idx and then idx to
  embedding) which could reduce embedding size if multiple words map to the
  same idx; although this is not currently a real or useful use-case.

  Args:
    embeddings_path: Local, GCS, or HDFS path to embedding file. Each line
      should be a word and its vector representation separated by a space.

  Returns:
    Tuple of:
      A vocabulary dictionary (mapping words to their index)
      A Numpy array of word embeddings with shape (vocab size, embedding size)
      A unique unknown token index (greater than all other token indexes)
      The size of the embeddings for words that is being used
  """
  word_to_idx = {}
  word_embeddings = []

  if not tf.gfile.Exists(embeddings_path):
    raise ValueError('File at %s does not exist.' % embeddings_path)

  with tf.gfile.Open(embeddings_path) as f:
    for idx, line in enumerate(f):
      values = line.split()
      word = values[0]
      word_embedding = np.asarray(values[1:], dtype='float32')
      word_to_idx[word] = idx + 1  # Reserve first row for padding
      word_embeddings.append(word_embedding)

  if not word_embeddings:
    raise ValueError('No embeddings loaded from %s.' % embeddings_path)

  # Add the padding "embedding"
  word_embeddings.insert(0, np.random.randn(len(word_embeddings[0])))

  # Convert embedding to numpy array and append the unknown word embedding,
  # which is the mean of all other embeddings.
  unknown_token = len(word_embeddings)
  embeddings_matrix = np.asarray(word_embeddings, dtype=np.float32)
  embeddings_matrix = np.append(
      embeddings_matrix, [embeddings_matrix.mean(axis=0)], axis=0)

  return word_to_idx, embeddings_matrix, unknown_token, len(word_embeddings[0])
