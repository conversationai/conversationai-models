# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Defines the dataset structure for evaluation pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import pandas as pd
from tensorflow.python.platform import tf_logging as logging

import utils_export.utils_cloudml as utils_cloudml


class Model(object):
  """Defines the spec of a CMLE Model.

    All models (given by `model_names`) need to share the feature_keys_spec,
      example_key and prediction_keys.
    Those fields define the inputs (feature_keys_spec, example_key) and output
      of the models.
    """

  def __init__(self,
               feature_keys_spec,
               prediction_keys,
               model_names,
               project_name,
               example_key='example_key'):
    """Initializes a model.

    Args:
      feature_keys_spec: spec of the tf_records input to the model.
      prediction_keys: Name of the keys to extract from model outputs.
      model_names: List of names of the model in Cloud MLE.
        Format should be $MODEL_NAME:$VERSION.
        If no version given, will take default version.
      project_name: name of the gcp project.
      example_key: name of the example key expected by the model.
        This example key will be created by the dataset function.

    Raises:
      ValueError: If example_key is included in the feature_spec
        of if feature_keys_spec does not match required format.
    """

    utils_cloudml.is_valid_spec(feature_keys_spec)
    if example_key in feature_keys_spec:
      raise ValueError('example_key should not be part of input_data.'
                       'It will be created when writing to tf-records')

    self._model_name = model_names
    self._feature_keys_spec = feature_keys_spec
    self._prediction_keys = prediction_keys
    self._project_name = project_name
    self._example_key = example_key

  def feature_keys_spec(self):
    return self._feature_keys_spec

  def example_key(self):
    return self._example_key

  def model_names(self):
    return self._model_name

  def prediction_keys(self):
    return self._prediction_keys

  def project_name(self):
    return self._project_name

  def set_job_ids_prediction(self, job_ids):
    self._job_ids_prediction = job_ids

  def job_ids_prediction(self):
    return self._job_ids_prediction


class Dataset(object):
  """Defines a format for every dataset to work with evaluation pipeline.

  Usage:

  input_fn = ... (returns pandas DataFrame).
  dataset = Dataset(input_fn) # Verifies that input_fn is ok.

  dataset.loads_data(10000, 0.5)

  model = Model(...)
  # Next function verifies that models are compatible.
  dataset.add_model_prediction_to_data(model)

  dataset.show_data()
  """

  def __init__(self, input_fn):
    self.check_input_fn(input_fn)
    self._input_fn = input_fn

  def show_data(self):
    if not hasattr(self, 'data'):
      raise ValueError(
          'Dataset does not have data yet.'
          ' You need to run `loads_data` first.')
    return self.data

  def check_input_fn(self, input_fn):
    """Checks if the input_fn meets requirements."""
    args_input_fn = inspect.getargspec(input_fn).args
    if 'max_n_examples' not in args_input_fn:
      raise ValueError(
          'input_fn should have (at least) `max_n_examples`'
          ' as arguments.')

    loaded_data = input_fn(max_n_examples=1)

    if not isinstance(loaded_data, pd.DataFrame):
      raise ValueError('input_fn should return a pandas DataFrame.')

    if len(loaded_data) != 1:
      raise ValueError(
          'input_fn(max_n_examples=1) should contain 1 row (exactly).')
    logging.info('input_fn is compatible with the `Dataset` class.')

  def check_compatibility(self, model):
    """Checks that input_fn is compatible with the model."""

    if hasattr(self, 'data'):
      test_df = self.data
    else:
      test_df = self._input_fn(max_n_examples=1)

    for key in model.feature_keys_spec():
      if key not in test_df.columns:
        raise ValueError(
            'input_fn must contain at least the feature keys {}'.format(
                model.feature_keys_spec()))
    logging.info('Model is compatible with the `Dataset` instance.')

  def load_data(self, max_n_examples, **kwargs):
    self.data = self._input_fn(max_n_examples, **kwargs)

  def get_path_prediction_tf(self, model_name):
    return self.tf_records_path.replace(
        '.tfrecords',
        '_predictions_{}'.format(model_name)
        )

  def convert_data_to_tf(self,
                         tf_records_path,
                         feature_keys_spec,
                         example_key,
                         overwrite=True):
    """Writes the data of a dataframe to tf-records.

    Args:
      tf_records_path: where to save the tf-records.
      feature_keys_spec: the spec of the feature_keys.
        Only those fields will be written to tf-records.
      example_key: Name of the field for example_key.
        The key will be generated on the fly.
      overwrite: Whether to overwrite the existing tf_records.

    Raises:
      ValueError: if dataset does not have data loaded.
    """

    if not hasattr(self, 'data'):
      raise ValueError(
          'Dataset does not have data yet.'
          ' You need to run `loads_data` first.')

    if hasattr(self, 'tf_records_path'):
      if overwrite:
        logging.info('TF-Records already exist - overwriting them.')
      else:
        logging.info('TF-Records already exist - We will use those.')
        return

    utils_cloudml.convert_pandas_to_tfrecords(
        self.data,
        feature_keys_spec,
        example_key,
        tf_records_path)
    self.tf_records_path = tf_records_path

  def call_prediction(self, model):
    """Starts a CMLE batch prediction job for the model."""

    if not hasattr(self, 'tf_records_path'):
      raise ValueError(
          'Dataset does not have tf_records_path yet.'
          ' You need to run `convert_data_to_tf` first.')

    job_ids = []
    for model_name_full in model.model_names():

      model_name_split = model_name_full.split(':')
      model_name = model_name_split[0]
      if len(model_name_split) > 1:
        version = model_name_split[1]
      else:
        version = None

      tf_record_tmp = self.get_path_prediction_tf(model_name_full)
      job_id = utils_cloudml.call_model_predictions_from_df(
          project_name=model.project_name(),
          tmp_tfrecords_gcs_path=self.tf_records_path,
          tmp_tfrecords_with_predictions_gcs_path=tf_record_tmp,
          model_name=model_name,
          version_name=version
      )
      job_ids.append(job_id)
    model.set_job_ids_prediction(job_ids)

  def collect_prediction(self, model):
    """Collected the predictions of CMLE jobs and adds it to dataframe."""

    if not hasattr(model, 'job_ids_prediction'):
      raise ValueError(
          'Model does not have any `job_ids_prediction`.'
          ' You need to run `call_prediction` for CMLE batch prediction job.')

    for model_name, job_id in zip(model.model_names(), model.job_ids_prediction()):
      tf_record_tmp = self.get_path_prediction_tf(model_name)
      self.data = utils_cloudml.add_model_predictions_to_df(
          job_id,
          self.data,
          project_name=model.project_name(),
          tmp_tfrecords_with_predictions_gcs_path=tf_record_tmp,
          column_name_of_model=model_name,
          prediction_name=model.prediction_keys(),
          example_key=model.example_key()
      )

  def add_model_prediction_to_data(self, model, tf_record_path_pattern):
    """Computes the prediction of the model and adds it to dataframe.

    Args:
      model: a `Model` instance
      tf_record_path_pattern: pattern of the path where to save the temporary
        files, in particular tf_records inputs and outputs of CMLE.
    """

    self.check_compatibility(model)

    tf_record_input_path = tf_record_path_pattern + '.tf_records'
    self.convert_data_to_tf(
        tf_record_input_path,
        model.feature_keys_spec(),
        model.example_key(),
        overwrite=True)

    self.call_prediction(model)
    self.collect_prediction(model)
