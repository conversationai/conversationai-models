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
"""Defines some utilities to use cloud MLE batch prediction jobs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import os
import re
import time

import googleapiclient.discovery as discovery
import googleapiclient.errors as errors
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_list_feature(value_list):
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(
          value=[tf.compat.as_bytes(value) for value in value_list]))


class FeatureSpec(object):

  INTEGER = 'integer'
  STRING = 'string'
  LIST_STRING = 'list_string'

  CONSTRUCTOR_PER_TYPE = {
      INTEGER: _int64_feature,
      STRING: _bytes_feature,
      LIST_STRING: _bytes_list_feature
  }


def is_valid_spec(spec):
  """Verfies that the spec matches requirements."""
  if not isinstance(spec, dict):
    raise ValueError('Spec should be a dictionary instance.')
  for (key, item) in spec.items():
    if not isinstance(key, str):
      raise ValueError(
          'Spec is badly defined. Keys should be string (field names).')
    if item not in FeatureSpec.CONSTRUCTOR_PER_TYPE.keys():
      raise ValueError(
          'Spec is badly defined. Authorized types are one of {}.'.format(
              FeatureSpec.CONSTRUCTOR_PER_TYPE.keys()))


def convert_pandas_to_tfrecords(df, feature_keys_spec, example_key, gcs_path):
  """Write a pandas `DataFrame` to a tf_record.

  Args:
    df: pandas `DataFrame`. It must include the fields that
      are part of feature_key_spec.
    feature_keys_spec: Dict of {name: type}, which describes the spec of the
      TF-records.
    example_key: key identifier of an example (string).
      This key will be added to data automatically and should not be
      part of df.
    gcs_path: where to write the tf records.
  Note: TFRecords will have fields feature_keys_spec and `example_key`.
  """

  writer = tf.python_io.TFRecordWriter(gcs_path)
  for i in range(len(df)):

    if not i % 10000:
      logging.info('Preparing train data: {}/{}'.format(i, len(df)))

    # Create a feature
    feature_dict = {}
    for feature in feature_keys_spec:
      constructor = FeatureSpec.CONSTRUCTOR_PER_TYPE[feature_keys_spec[feature]]
      feature_dict[feature] = constructor(df[feature].iloc[i])
      feature_dict[example_key] = _int64_feature(i)
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

  writer.close()


def call_model_predictions_from_df(project_name,
                                   tmp_tfrecords_gcs_path,
                                   tmp_tfrecords_with_predictions_gcs_path,
                                   model_name,
                                   version_name=None):
  """Calls a prediction job.

  Args:
    project_name: gcp project name.
    tmp_tfrecords_gcs_path: gcs path to store tf_records, which will be inputs
      to batch prediction job.
    tmp_tfrecords_with_predictions_gcs_path: gcs path to store tf_records, which
      will be outputs to batch prediction job.
    model_name: Model name used to run predictions. The model must take as
      inputs TF-Records with fields $TEXT_FEATURE_NAME and $SENTENCE_KEY, and
      should return a dictionary including the field $LABEL_NAME.
    version_name: Model version to run predictions. If None, it will use default
      version of the model.

  Returns:
    job_id: the job_id of the prediction job.

  Raises:
    ValueError: if tmp_tfrecords_gcs_path does not exist.
  """

  # Create tf-records if necessary.
  if not file_io.file_exists(tmp_tfrecords_gcs_path):
    raise ValueError('tf_records do not exist.')

  # Call batch prediction job.
  job_id = _call_batch_job(
      project_name,
      input_paths=tmp_tfrecords_gcs_path,
      output_path=tmp_tfrecords_with_predictions_gcs_path,
      model_name=model_name,
      version_name=version_name)

  return job_id


def add_model_predictions_to_df(
    job_id, df, project_name, tmp_tfrecords_with_predictions_gcs_path,
    column_name_of_model, prediction_name, example_key):
  """Add predictions from a cloud prediction job to the pandas dataframe.

  Args:
    job_id: the job_id of the prediction job.
    df: a pandas `DataFrame`.
    project_name: gcp project name.
    tmp_tfrecords_with_predictions_gcs_path: gcs path to store tf_records, which
      will be outputs to batch prediction job.
    column_name_of_model: Name of the added column.
    prediction_name: Name of the column to retrieve from CMLE predictions.
    example_key: key identifier of an example.

  Returns:
    df: a pandas ` DataFrame` with an added column named 'column_name_of_model'
      containing the prediction values.
  """

  # Waits for batch job to be over.
  _check_job_over(project_name, job_id)

  # Add one prediction column to the database.
  tf_records_path = os.path.join(tmp_tfrecords_with_predictions_gcs_path,
                                 'prediction.results-00000-of-00001')
  df_with_predictions = _combine_prediction_to_df(
      df, tf_records_path, column_name_of_model, prediction_name, example_key)

  return df_with_predictions


def _make_batch_job_body(project_name,
                         input_paths,
                         output_path,
                         model_name,
                         region='us-central1',
                         data_format='TF_RECORD',
                         version_name=None,
                         max_worker_count=None,
                         runtime_version=None):
  """Creates the request body for Cloud MLE batch prediction job."""

  project_id = 'projects/{}'.format(project_name)
  model_id = '{}/models/{}'.format(project_id, model_name)
  if version_name:
    version_id = '{}/versions/{}'.format(model_id, version_name)

  # Make a jobName of the format "model_name_batch_predict_YYYYMMDD_HHMMSS"
  timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())

  # Make sure the project name is formatted correctly to work as the basis
  # of a valid job name.
  clean_project_name = re.sub(r'\W+', '_', project_name)

  job_id = '{}_{}_{}'.format(clean_project_name, model_name, timestamp)

  # Start building the request dictionary with required information.
  body = {
      'jobId': job_id,
      'predictionInput': {
          'dataFormat': data_format,
          'inputPaths': input_paths,
          'outputPath': output_path,
          'region': region
      }
  }

  # Use the version if present, the model (its default version) if not.
  if version_name:
    body['predictionInput']['versionName'] = version_id
  else:
    body['predictionInput']['modelName'] = model_id

  # Only include a maximum number of workers or a runtime version if specified.
  # Otherwise let the service use its defaults.
  if max_worker_count:
    body['predictionInput']['maxWorkerCount'] = max_worker_count

  if runtime_version:
    body['predictionInput']['runtimeVersion'] = runtime_version

  return body


def _call_batch_job(project_name,
                    input_paths,
                    output_path,
                    model_name,
                    version_name=None):
  """Calls a batch prediction job on Cloud MLE."""

  batch_predict_body = _make_batch_job_body(
      project_name,
      input_paths,
      output_path,
      model_name,
      version_name=version_name)

  project_id = 'projects/{}'.format(project_name)

  ml = discovery.build('ml', 'v1')
  request = ml.projects().jobs().create(
      parent=project_id, body=batch_predict_body)

  try:
    response = request.execute()
    logging.info('state : {}'.format(response['state']))
    return response['jobId']

  except errors.HttpError as err:
    # Something went wrong, print out some information.
    logging.info('There was an error getting the prediction results.'
                 'Check the details:')
    logging.info(err._get_reason())


def _check_job_over(project_name, job_name):
  """Sleeps until the batch job is over."""

  clean_project_name = re.sub(r'\W+', '_', project_name)

  ml = discovery.build('ml', 'v1')
  request = ml.projects().jobs().get(
      name='projects/{}/jobs/{}'.format(clean_project_name, job_name))

  job_completed = False
  k = 0
  start_time = datetime.datetime.now()
  while not job_completed:
    response = request.execute()
    job_completed = (response['state'] == 'SUCCEEDED')
    if not job_completed:
      if not (k % 5):
        time_spent = int((datetime.datetime.now() - start_time).total_seconds() / 60)
        logging.info(
            'Waiting for prediction job to complete. Minutes elapsed: {}'
            .format(time_spent))
      time.sleep(30)
    k += 1

  logging.info('Prediction job completed.')


def _combine_prediction_to_df(df, prediction_file, model_col_name,
                              prediction_name, example_key):
  """Loads the prediction files and adds the model scores to a DataFrame.

  The dataframe and the prediction file must correspond exactly (same number
    of lines and same order)."""

  def load_predictions(prediction_file):
    with file_io.FileIO(prediction_file, 'r') as f:
      # prediction file needs to fit in memory.
      predictions = [json.loads(line) for line in f]
    return predictions

  predictions = load_predictions(prediction_file)
  predictions = sorted(predictions, key=lambda x: x[example_key])

  if len(predictions) != len(df):
    raise ValueError(
        'The dataframe and the prediction file do not contain'
        ' the same number of lines.'
    )

  prediction_proba = [x[prediction_name][0] for x in predictions]

  df[model_col_name] = prediction_proba

  return df
