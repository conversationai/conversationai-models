"""Preprocessing steps of the data preparation."""

import os
import random

import apache_beam as beam
import tensorflow as tf
from tensorflow_transform import coders


def get_identity_list():
  return ['male', 'female', 'transgender', 'other_gender', 'heterosexual',
          'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation',
          'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist',
          'other_religion', 'black', 'white', 'asian', 'latino',
          'other_race_or_ethnicity', 'physical_disability',
          'intellectual_or_learning_disability',
          'psychiatric_or_mental_illness', 'other_disability']


def get_civil_comments_spec(include_identity_terms=True):
  """Returns the spec of the civil_comments dataset."""
  spec = {
      'comment_text': tf.FixedLenFeature([], dtype=tf.string),
      'id': tf.FixedLenFeature([], dtype=tf.string),
      'toxicity': tf.FixedLenFeature([], dtype=tf.float32),
      'severe_toxicity': tf.FixedLenFeature([], dtype=tf.float32),
      'obscene': tf.FixedLenFeature([], dtype=tf.float32),
      'sexual_explicit': tf.FixedLenFeature([], dtype=tf.float32),
      'identity_attack': tf.FixedLenFeature([], dtype=tf.float32),
      'insult': tf.FixedLenFeature([], dtype=tf.float32),
      'threat': tf.FixedLenFeature([], dtype=tf.float32),
      'toxicity_annotator_count': tf.FixedLenFeature([], dtype=tf.int64),
      'identity_annotator_count': tf.FixedLenFeature([], dtype=tf.int64),
  }
  if include_identity_terms:
    for identity in get_identity_list():
      spec[identity] = tf.FixedLenFeature(
          [], dtype=tf.float32, default_value=-1.0)
  return spec


class Schema(object):
  """Defines the dataset schema for tf-transform.

  We should have used dataset_schema from tensorflow_transform.tf_metadata.
      However, there is a lack of support for `FixedLenFeature` default value,
      and an exception is triggered by _feature_from_feature_spec.
  TODO(fprost): Submit internal bug here.
  """

  def __init__(self, spec):
    self._spec = spec

  def as_feature_spec(self):
    return self._spec


class DecodeTFRecord(beam.DoFn):
  """Wrapper around ExampleProtoCoder for decoding optional fields.

  To decode a TF-Record example, we use the  coder utility
    'tensorflow_transform.codersExampleProtoCoder'. For optional fields,
    (indicated by 'default_value' argument for `FixedLenFeature`), the coder
    will generate the default value when the optional field is missing.
  This wrapper post-processes the coder and removes the field if the default
      value was used.
  """

  def __init__(self,
               feature_spec,
               optional_field_names,
               rule_optional_fn=lambda x: x < 0):
    """Initialises a TF-Record decoder.

    Args:
      feature_spec: Dictionary from feature names to one of
          `FixedLenFeature`, `SparseFeature` or `VarLenFeature.
          It contains all the features to parse (including optional ones).
      optional_field_names: list of optional fields.
      rule_optional_fn: function that take the value of an optional field
          and returns True if the value is indicative of a default value
          (e.g. resulting from the default value of parsing FixedLenFeature).

    Current code requires that all optional_field_names share the
        rule_optional_fn.
    """
    self._schema = Schema(feature_spec)
    self._coder = coders.ExampleProtoCoder(self._schema)
    self._optional_field_names = optional_field_names
    self._rule_optional_fn = rule_optional_fn

  def process(self, element):
    parsed_element = self._coder.decode(element)
    for identity in self._optional_field_names:
      if self._rule_optional_fn(parsed_element[identity]):
        del parsed_element[identity]
    yield parsed_element


class EncodeTFRecord(beam.DoFn):
  """Wrapper around ExampleProtoCoder for encoding optional fields."""

  def __init__(self, feature_spec, optional_field_names):
    """Initialises a TF-Record encoder.

    Args:
      feature_spec: Dictionary from feature names to one of
          `FixedLenFeature`, `SparseFeature` or `VarLenFeature.
          It contains all the features to parse (including optional ones).
      optional_field_names: list of optional fields.
    """
    self._feature_spec = feature_spec
    self._optional_field_names = optional_field_names

  def process(self, element):
    element_spec = self._feature_spec.copy()
    for identity in self._optional_field_names:
      if identity not in element:
        del element_spec[identity]
    element_schema = Schema(element_spec)
    coder = coders.ExampleProtoCoder(element_schema)
    encoded_element = coder.encode(element)
    yield encoded_element


def split_data(examples, train_fraction, eval_fraction):
  """Splits the data into train/eval/test."""

  def partition_fn(data, n_partition):
    random_value = random.random()
    if random_value < train_fraction:
      return 0
    if random_value < train_fraction + eval_fraction:
      return 1
    return 2
  examples_split = (examples
                    | 'SplitData' >> beam.Partition(partition_fn, 3))
  return examples_split


class OversampleExample(beam.DoFn):
  """Oversamples examples from a given class."""

  def __init__(self, rule_fn, oversample_rate):
    if (oversample_rate <= 0) or not isinstance(oversample_rate, int):
      raise ValueError('oversample_rate should be a positive integer.')
    self._rule_fn = rule_fn
    self._oversample_rate = oversample_rate

  def process(self, element):
    if self._rule_fn(element):
      for _ in range(self._oversample_rate):
        yield element
    else:
      yield element


def select_male_toxic_example(example,
                              threshold_identity=0.5,
                              threshold_toxic=0.5):
  is_toxic = example['toxicity'] >= threshold_toxic
  if 'male' in example:
    is_male = example['male'] >= threshold_identity
  else:
    is_male = False
  return is_toxic and is_male


@beam.ptransform_fn
def Shuffle(examples):  # pylint: disable=invalid-name
  return (examples
          | 'PairWithRandom' >> beam.Map(lambda x: (random.random(), x))
          | 'GroupByRandom' >> beam.GroupByKey()
          | 'DropRandom' >> beam.FlatMap(lambda (k, vs): vs))


def write_to_tf_records(examples, output_path):
  """Shuffles and writes to disk."""

  output_path_prefix = os.path.basename(output_path)
  shuff_ex = (examples | 'Shuffle_' + output_path_prefix >> Shuffle())
  _ = (shuff_ex
       | 'Serialize_' + output_path_prefix >> beam.ParDo(
           EncodeTFRecord(
               feature_spec=get_civil_comments_spec(), \
               optional_field_names=get_identity_list()))
       | 'WriteToTF_' + output_path_prefix >> beam.io.WriteToTFRecord(
           file_path_prefix=output_path,
           file_name_suffix='.tfrecord'))


def run(p, input_data_path, train_fraction, eval_fraction, output_folder):
  """Runs the data processing."""
  examples = (p
              | 'ReadExamples' >> beam.io.tfrecordio.ReadFromTFRecord(
                  file_pattern=input_data_path))
  examples = (examples
              | 'DecodeTFRecord' >> beam.ParDo(DecodeTFRecord(
                  feature_spec=get_civil_comments_spec(),
                  optional_field_names=get_identity_list())))

  split = split_data(examples, train_fraction, eval_fraction)
  train_data = split[0]
  eval_data = split[1]
  test_data = split[2]

  write_to_tf_records(
      train_data,
      os.path.join(output_folder, 'train_data'))
  write_to_tf_records(
      eval_data,
      os.path.join(output_folder, 'eval_data'))
  write_to_tf_records(
      test_data,
      os.path.join(output_folder, 'test_data'))

  train_data_artificially_biased = (
      train_data
      | 'CreateBias' >> beam.ParDo(OversampleExample(
          select_male_toxic_example, 2)))

  write_to_tf_records(
      train_data_artificially_biased,
      os.path.join(output_folder, 'train_data_artificially_biased'))
