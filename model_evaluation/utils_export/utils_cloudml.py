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


def call_model_predictions_from_df(project_name,
                                   input_tf_records,
                                   output_prediction_path,
                                   model_name,
                                   version_name=None):
  """Calls a prediction job.

  Args:
    project_name: gcp project name.
    input_tf_records: gcs path to input tf_records.
    output_prediction_path: gcs path to store tf_records, which will be outputs
      to batch prediction job.
    model_name: Model name used to run predictions. The model must take as
      inputs TF-Records with fields $TEXT_FEATURE_NAME and $SENTENCE_KEY, and
      should return a dictionary including the field $LABEL_NAME.
    version_name: Model version to run predictions. If None, it will use default
      version of the model.

  Returns:
    job_id: the job_id of the prediction job.

  Raises:
    ValueError: if input_tf_records does not exist.
  """

  # Create tf-records if necessary.
  if not file_io.file_exists(input_tf_records):
    raise ValueError('tf_records do not exist.')

  # Call batch prediction job.
  job_id = _call_batch_job(
      project_name,
      input_paths=input_tf_records,
      output_path=output_prediction_path,
      model_name=model_name,
      version_name=version_name)

  return job_id


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


def check_job_over(project_name, job_name):
  """Sleeps until the batch job is over."""

  #TODO(nthain): Why did we use this?
  #clean_project_name = re.sub(r'\W+', '_', project_name)
  clean_project_name = project_name

  ml = discovery.build('ml', 'v1')
  request = ml.projects().jobs().get(
      name='projects/{}/jobs/{}'.format(clean_project_name, job_name))
  #print('DEBUG OUTPUT: request name')
  #print('projects/{}/jobs/{}'.format(clean_project_name, job_name))
  job_completed = False
  k = 0
  start_time = datetime.datetime.now()
  while not job_completed:
    response = request.execute()
    job_completed = (response['state'] == 'SUCCEEDED')
    if not job_completed:
      if not (k % 5):
        time_spent = int(
            (datetime.datetime.now() - start_time).total_seconds() / 60)
        logging.info(
            'Waiting for prediction job to complete. Minutes elapsed: {}'
            .format(time_spent))
      time.sleep(30)
    k += 1

  logging.info('Prediction job completed.')


def add_model_predictions_to_df(df, prediction_file, model_col_name,
                                prediction_name, example_key, class_names):
  """Loads the prediction files and adds the model scores to a DataFrame.

  Args:
    df: a pandas `DataFrame`.
    prediction_file: Path to the prediction files (outputs of CMLE prediction
      job).
    model_col_name: Column name of the prediction values in df (added column).
    prediction_name: Name of the column to retrieve from CMLE predictions.
    example_key: key identifier of an example.
    class_names: If the model is a multiclass model, you can specify class names.
          The model will then return a logit value per class instead of a single value.
  Returns:
    df: a pandas ` DataFrame` with an added column named 'column_name_of_model'
      containing the prediction values.

  Raises:
    ValueError: dataframe and  prediction file do not correspond exactly
      In particular, they must have same number of lines and same order.
    ValueError: prediction file does not exist.

  This function reads the prediction file and extracts the fields
  'prediction_name'
    and example_key. It orders the results based on example_key and then adds
    them to df
    in a new column called 'model_col_name'.
  """

  prediction_file = os.path.join(prediction_file,
                                 'prediction.results-00000-of-00001')
  if not tf.gfile.Exists(prediction_file):
    raise ValueError(
        'Prediction file does not exist.'
        ' You need to call prediction job and wait for completion.')

  def _load_predictions(pred_file):
    with file_io.FileIO(pred_file, 'r') as f:
      # prediction file needs to fit in memory.
      try:
        predictions = [json.loads(line) for line in f]
      except:
        predictions = []
    return predictions

  predictions = _load_predictions(prediction_file)

  if not predictions:
    raise ValueError(
        'The prediction file returned by CMLE is empty.'
        ' It might be due to a badly formatted tfrecord input file that can not be'
        ' parsed by CMLE (wrong input signature given by a `Model` instance).'
        ' Check the logs of your CMLE job for further details.')
  if example_key not in predictions[0]:
    raise ValueError(
        "Predictions do not contain the 'example_key' field."
        " Verify that your 'example_key' parameter (set to {})"
        " matches the CMLE model signature.".format(example_key))
  if prediction_name not in predictions[0]:
    raise ValueError(
        "Predictions do not contain the 'prediction_name' field."
        " Verify that your 'prediction_name' parameter (set to {})"
        " matches the CMLE model signature.".format(prediction_name))
  if len(predictions) != len(df):
    raise ValueError('The dataframe and the prediction file do not contain'
                     ' the same number of lines.')

  predictions = sorted(predictions, key=lambda x: x[example_key])
  if class_names is None:
      prediction_proba = [x[prediction_name][0] for x in predictions]
      df[model_col_name] = prediction_proba
  else:
      for i, class_name in enumerate(class_names):
            df['{}_{}'.format(model_col_name,class_name)] = [x[prediction_name][i] for x in predictions]

  return df
