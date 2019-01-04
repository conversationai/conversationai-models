"""Preprocessing steps of the data preparation."""

import os
import random

import apache_beam as beam
import tensorflow as tf
from tensorflow_transform import coders

import constants
import tfrecord_utils


def get_identity_list():
  return [
      'male', 'female', 'transgender', 'other_gender', 'heterosexual',
      'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation',
      'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist',
      'other_religion', 'black', 'white', 'asian', 'latino',
      'other_race_or_ethnicity', 'physical_disability',
      'intellectual_or_learning_disability', 'psychiatric_or_mental_illness',
      'other_disability'
  ]


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
      spec[identity] = tf.FixedLenFeature([],
                                          dtype=tf.float32,
                                          default_value=-1.0)
  return spec


def split_data(examples, train_fraction, eval_fraction):
  """Splits the data into train/eval/test."""

  def partition_fn(data, n_partition):
    random_value = random.random()
    if random_value < train_fraction:
      return 0
    if random_value < train_fraction + eval_fraction:
      return 1
    return 2

  examples_split = (examples | 'SplitData' >> beam.Partition(partition_fn, 3))
  return examples_split


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
  _ = (
      shuff_ex
      | 'Serialize_' + output_path_prefix >> beam.ParDo(
          tfrecord_utils.EncodeTFRecord(
              feature_spec=get_civil_comments_spec(),
              optional_field_names=get_identity_list()))
      | 'WriteToTF_' + output_path_prefix >> beam.io.WriteToTFRecord(
          file_path_prefix=output_path, file_name_suffix='.tfrecord'))


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


def _select_male_toxic_example(example,
                               threshold_identity=0.5,
                               threshold_toxic=0.5):
  is_toxic = example['toxicity'] >= threshold_toxic
  if 'male' in example:
    is_male = example['male'] >= threshold_identity
  else:
    is_male = False
  return is_toxic and is_male


def run_data_split(p, input_data_path, train_fraction, eval_fraction,
                   output_folder):
  """Splits the data into train/eval/test.

  Args:
    p: Beam pipeline for constructing PCollections and applying PTransforms.
    input_data_path: Input TF Records.
    train_fraction: Fraction of the data to be allocated to the training set.
    eval_fraction: Fraction of the data to be allocated to the eval set.
    output_folder: Folder to save the train/eval/test datasets.

  Raises:
    ValueError:
        If train_fraction + eval_fraction >= 1.
        If the output_directory exists. This exception prevents the user
            from overwriting a previous split.
  """

  if (train_fraction + eval_fraction >= 1.):
    raise ValueError('Train and eval fraction are incompatible.')
  if tf.gfile.Exists(output_folder):
    raise ValueError('Output directory should be empty.'
                     ' You should select a different path.')

  examples = (
      p
      | 'ReadExamples' >>
      beam.io.tfrecordio.ReadFromTFRecord(file_pattern=input_data_path))
  examples = (
      examples
      | 'DecodeTFRecord' >> beam.ParDo(
          tfrecord_utils.DecodeTFRecord(
              feature_spec=get_civil_comments_spec(),
              optional_field_names=get_identity_list())))

  split = split_data(examples, train_fraction, eval_fraction)
  train_data = split[0]
  eval_data = split[1]
  test_data = split[2]

  write_to_tf_records(train_data,
                      os.path.join(output_folder, constants.TRAIN_DATA_PREFIX))
  write_to_tf_records(eval_data,
                      os.path.join(output_folder, constants.EVAL_DATA_PREFIX))
  write_to_tf_records(test_data,
                      os.path.join(output_folder, constants.TEST_DATA_PREFIX))


def run_artificial_bias(p, train_input_data_path, output_folder,
                        oversample_rate):
  """Main function to create artificial bias.

  Args:
    p: Beam pipeline for constructing PCollections and applying PTransforms.
    train_input_data_path: Input TF Records, which is typically the training
      dataset. This artificial bias method should not be run on eval/test.
    output_folder: Folder to save the train/eval/test datasets.
    oversample_rate: How many times to oversample the targeted class.
  """

  train_data = (
      p
      | 'ReadExamples' >>
      beam.io.tfrecordio.ReadFromTFRecord(file_pattern=train_input_data_path)
      | 'DecodeTFRecord' >> beam.ParDo(
          tfrecord_utils.DecodeTFRecord(
              feature_spec=get_civil_comments_spec(),
              optional_field_names=get_identity_list())))

  train_data_artificially_biased = (
      train_data
      | 'CreateBias' >> beam.ParDo(
          OversampleExample(_select_male_toxic_example, oversample_rate)))

  write_to_tf_records(
      train_data_artificially_biased,
      os.path.join(output_folder, constants.TRAIN_ARTIFICIAL_BIAS_PREFIX))
