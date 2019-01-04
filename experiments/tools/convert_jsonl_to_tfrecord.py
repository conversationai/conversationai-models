# coding=utf-8
# Copyright 2018 The Conversation-AI.github.io Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A function to convert jsonlines to TFRecords.

python tools/convert_jsonl_to_tfrecord.py \
 --input_jsonlines_path=tf_trainer/common/testdata/cats_and_dogs.jsonl \
 --text_fields_re='^(text)$' \
 --label_fields_re='^(bad)$' \
 --output_tfrecord_path=local_data/testdata/cats_and_dogs.recordio
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app
from absl import logging
import json
import jsonlines
import tensorflow as tf
import re

FLAGS = flags.FLAGS

# TODO: Compute basic stats for text fields and labels.
flags.DEFINE_string('text_fields_re', None,
                    'Matcher for names of the text fields.')
flags.register_validator(
    'text_fields_re',
    lambda value: isinstance(value, str) and re.compile(value),
    message='--text_field_re must be a regexp string.')

flags.DEFINE_string('label_fields_re', None,
                    'Matcher for names of the label fields.')
flags.register_validator(
    'label_fields_re',
    lambda value: isinstance(value, str) and re.compile(value),
    message='--label_fields_re must be a regexp string.')

flags.DEFINE_string('input_jsonlines_path', None,
                    'Path to the JSON-lines input file.')
flags.register_validator(
    'input_jsonlines_path',
    lambda value: isinstance(value, str),
    message='--input_jsonlines_path must be a string.')

flags.DEFINE_string('output_tfrecord_path', None,
                    'Path where the output TFRecord should be written.')
flags.register_validator(
    'output_tfrecord_path',
    lambda value: isinstance(value, str),
    message='--output_tfrecord_path must be a string.')


class MisingAllTextFieldsError(Exception):
  pass


class FieldsCounter():

  def __init__(self):
    self.counters = {}

  def inc_field(self, field_name: str):
    if field_name not in self.counters:
      self.counters[field_name] = 0
    self.counters[field_name] += 1


def make_selected_output_row(row, line, counters):
  """Create an output row with just the fields matching --text_fields_re and

  --label_fields_re. If there is no matching field in the row for
  --text_fields_re then raise MisingAllTextFieldsError.
  """
  text_field_matcher = re.compile(FLAGS.text_fields_re)
  label_field_matcher = re.compile(FLAGS.label_fields_re)
  has_text_field = False
  output_row = {}
  for key, value in row.items():
    if text_field_matcher.match(key):
      has_text_field = True
      counters.inc_field(key)
      output_row[key] = value
    elif label_field_matcher.match(key):
      counters.inc_field(key)
      output_row[key] = value
  if not has_text_field:
    raise MisingAllTextFieldsError(
        f'Error parsing file {input_jsonlines_path} at line: {line}.\n'
        f'No field matched by --text_field_regexp="{FLAGS.text_fields_re}":\n'
        f'  {json.dumps(row, sort_keys=True, indent=2)}')
  return output_row


def itr_as_dict(input_jsonlines_path):
  with tf.gfile.Open(input_jsonlines_path) as f:
    counters = FieldsCounter()
    line = 0
    for row in jsonlines.Reader(f):
      line += 1
      yield make_selected_output_row(row, line, counters)
    logging.info(f'Complete.\nField Counts:\n'
                 f'{json.dumps(counters.counters, sort_keys=True, indent=2)}')


def itr_as_tfrecord(input_jsonlines_path):
  for row in itr_as_dict(input_jsonlines_path):
    example = tf.train.Example()
    for key, value in row.items():
      if isinstance(value, str):
        example.features.feature[key].bytes_list.value.append(
            value.encode('utf-8', errors='replace'))
      elif isinstance(value, float) or isinstance(value, int):
        example.features.feature[key].float_list.value.append(value)
    yield example


def convert_to_tfrecord(input_jsonlines_path, output_tfrecord_path):
  with tf.python_io.TFRecordWriter(output_tfrecord_path) as writer:
    for example in itr_as_tfrecord(input_jsonlines_path):
      writer.write(example.SerializeToString())


def main(argv):
  del argv  # unused
  convert_to_tfrecord(FLAGS.input_jsonlines_path, FLAGS.output_tfrecord_path)


if __name__ == '__main__':
  app.run(main)
