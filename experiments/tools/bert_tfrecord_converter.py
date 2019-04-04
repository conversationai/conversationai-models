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

"""Converts our TFRecord data into the format expected by the BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bert
from bert import run_classifier
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

tf.app.flags.DEFINE_string('input_data_path', None,
                           'Path to the input TFRecord files.')
tf.app.flags.DEFINE_string('output_data_path', None,
                           'Path to write the output TFRecord files.')
tf.app.flags.DEFINE_string('filenames', None,
                           'Comma separated list of filenames.')
tf.app.flags.DEFINE_string('bert_url', 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1', 'TF Hub URL for BERT Model')
tf.app.flags.DEFINE_integer('max_sequence_length', 256,
                            'Maximum sequence length of tokenized comment.')

FLAGS = tf.app.flags.FLAGS

def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

def create_tokenizer_from_hub_module(url):
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(url)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])

  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_tfrecord_for_bert(filenames,
                              input_data_path,
                              output_data_path,
                              bert_tfhub_url,
                              max_seq_length):
  """Converts input TFRecords into the format expected by the BERT model."""
  tokenizer = create_tokenizer_from_hub_module(bert_tfhub_url)
  for filename in filenames:
    print('Working on {}...'.format(filename))
    in_filepath = '{}{}'.format(input_data_path, filename)
    #TODO: Check if file exists, if not write new file
    #TODO: Have the filename reflect the max_sequence_length and path reflect model
    out_filepath = '{}{}'.format(output_data_path, filename)
    record_iterator = tf.python_io.tf_record_iterator(path=in_filepath)
    writer = tf.python_io.TFRecordWriter(out_filepath)
    for ex_index, string_record in enumerate(record_iterator):
      example = tf.train.Example()
      example.ParseFromString(string_record)
      text = example.features.feature[text_key].bytes_list.value[0]
      label = example.features.feature[label_key].float_list.value[0]
      label = round(label)
      ex = run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                      text_a = text, 
                                      text_b = None, 
                                      label = label)
      label_list = [0, 1]
      feature = run_classifier.convert_single_example(ex_index, ex, label_list,
                                                      max_seq_length, tokenizer)
      features = collections.OrderedDict()
      features["input_ids"] = create_int_feature(feature.input_ids)
      features["input_mask"] = create_int_feature(feature.input_mask)
      features["segment_ids"] = create_int_feature(feature.segment_ids)
      features["label_ids"] = create_int_feature([feature.label_id])
      features["is_real_example"] = create_int_feature(
          [int(feature.is_real_example)])

      tf_example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())
    writer.close()
    print('... Done!')

if __name__ == '__main__':
  filenames = [name.strip() for name in FLAGS.filenames.split(',')]
  convert_tfrecord_for_bert(filenames,
                            FLAGS.input_data_path,
                            FLAGS.output_data_path,
                            FLAGS.bert_url,
                            FLAGS.max_sequence_length)