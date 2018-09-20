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
"""Deploys all models that have been saved in a directory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import sys
import time

from googleapiclient import discovery
from googleapiclient import errors
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging


# Maximum number of version that can be created concurrently.
CLOUD_ML_VERSION_CREATE_QUOTA = 10


def get_list_models_to_export(parent_model_dir):
  """Gets the paths of all models that are in parent_model_dir."""
  _list = []
  for subdirectory, _, files in tf.gfile.Walk(parent_model_dir):
    if 'saved_model.pb' in files: # Indicator of a saved model.
      _list.append(subdirectory)
  return _list


def check_model_exists(project_name, model_name):
  """Verifies if a model name is deployed already on CMLE."""
  ml = discovery.build('ml', 'v1')

  model_id = 'projects/{}/models/{}'.format(project_name, model_name)
  request = ml.projects().models().get(
      name=model_id)
  try:
    response = request.execute()
    return True
  except:
    return False


def create_model(project_name, model_name):
  """Creates a model on CMLE."""
  ml = discovery.build('ml', 'v1')

  request_dict = {'name': model_name}
  project_id = 'projects/{}'.format(project_name)
  request = ml.projects().models().create(
      parent=project_id,
      body=request_dict)
  try:
    response = request.execute()
  except errors.HttpError as err:
    raise ValueError(
        'There was an error creating the model.' +
        ' Check the details: {}'.format(err._get_reason()))


def create_version(project_name, model_name, version_name, model_dir):
  """Creates a version of a model on CMLE."""

  ml = discovery.build('ml', 'v1')
  request_dict = {
      'name': version_name,
      'deploymentUri': model_dir,
      'runtimeVersion': '1.10'
      }
  model_id = 'projects/{}/models/{}'.format(project_name, model_name)
  request = ml.projects().models().versions().create(
      parent=model_id,
      body=request_dict)

  try:
    response = request.execute()
    operation_id = response['name']
    return operation_id

  except errors.HttpError as err:
    raise ValueError(
        'There was an error creating the version.' +
        ' Check the details:'.format(err._get_reason()))


def check_version_deployed(operation_id):
  """Loops until the version has been deployed on CMLE."""

  ml = discovery.build('ml', 'v1')
  request = ml.projects().operations().get(name=operation_id)

  done = False
  while not done:
    response = None
    time.sleep(0.3)
    try:
        response = request.execute()
        done = response.get('done', False)
    except errors.HttpError as err:
        raise ValueError(
        'There was an error getting the operation.' +
        ' Check the details: {}'.format(err._get_reason()))
        done = True


def deploy_model_version(project_name, model_name, version_name, model_dir):
  """Deploys one TF model on CMLE.

  Args:
    project_name: Name of a CMLE project.
    model_name: Name of the model to deploy.
      If it does not exist yet, the model will be created.
    version_name: Version of the model on CMLE.
    Model_dir: Where to find the exported model.
  """

  if not check_model_exists(project_name, model_name):
    create_model(project_name, model_name)
  operation_id = create_version(project_name, model_name, version_name, model_dir)
  return operation_id


def _get_version_name(model_dir):
  return  'v_{}'.format(os.path.basename(os.path.dirname(model_dir)))


def deploy_all_models(parent_dir, project_name, model_name):
  """Finds and deploys all models present in the directory.
  
  Args:
    parent_dir: Directory to explore.
    project_name: Name of the project.
    model_name: Name of the model.
      All the model found in the parent_dir will be 
      saved within the same main model.
  """

  list_models = get_list_models_to_export(parent_dir)
  logging.info('Exploration finished: {} models detected.'.format(len(list_models)))

  num_epochs = int(len(list_models) / CLOUD_ML_VERSION_CREATE_QUOTA)

  for i in range(0, num_epochs + 1):
    indices = range(i * CLOUD_ML_VERSION_CREATE_QUOTA, (i+1) * CLOUD_ML_VERSION_CREATE_QUOTA)
    operation_id_list = []
    for j in indices:
      if j >= len(list_models):
        break
      version_name = _get_version_name(list_models[j])
      operation_id = deploy_model_version(
          project_name=project_name,
          model_name=model_name,
          version_name=version_name,
          model_dir=list_models[j]
          )
      operation_id_list.append(operation_id)

    logging.info('Waiting for versions to be deployed...')
    for operation_id in operation_id_list:
      check_version_deployed(operation_id)

  logging.info('DONE. {} models have been deployed'.format(len(list_models)))


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--parent_dir',
      help='Name of the parent model directory.',
      default='gs://kaggle-model-experiments/tf_trainer_runs/fprost/tf_gru_attention/20180917_161053/model_dir/')
  parser.add_argument(
      '--project_name',
      help='Name of GCP project.',
      default='wikidetox')
  parser.add_argument(
      '--model_name',
      help='Name of the model on CMLE.',
      default='tf_gru_attention_continuous')
  args = parser.parse_args(args=sys.argv[1:])

  tf.logging.set_verbosity(tf.logging.INFO)

  deploy_all_models(args.parent_dir, args.project_name, args.model_name)
