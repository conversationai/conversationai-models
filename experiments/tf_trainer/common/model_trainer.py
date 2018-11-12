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
"""The Model Trainer class.

This provides an abstraction of Keras and TF.Estimator, and is intended for use
in text classification models (although it may generalize to other kinds of
problems).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import os.path
import six

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.export.export_output import PredictOutput
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.lib.io import file_io

from tf_trainer.common import base_model
from tf_trainer.common import dataset_input as ds


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_path', None,
                           'Path to the training data TFRecord file.')
tf.app.flags.DEFINE_string('validate_path', None,
                           'Path to the validation data TFRecord file.')
tf.app.flags.DEFINE_string('model_dir', None,
                           "Directory for the Estimator's model directory.")
tf.app.flags.DEFINE_bool('enable_profiling', False,
                         'Enable profiler hook in estimator.')
tf.app.flags.DEFINE_integer('n_export', 1,
                            'Number of models to export.'
                            'If =1, only the last one is saved.'
                            'If >1, we split the take n_export checkpoints.')

tf.app.flags.mark_flag_as_required('train_path')
tf.app.flags.mark_flag_as_required('validate_path')
tf.app.flags.mark_flag_as_required('model_dir')


# This function extends tf.contrib.estimator.forward_features.
# As the binary_head has a ClassificationOutput for serving_default,
# the check at the end of 'new_model_fn' fails in the initial fn.
def forward_features(estimator, keys, sparse_default_values=None):
  """Forward features to predictions dictionary.
  In some cases, user wants to see some of the features in estimators prediction
  output. As an example, consider a batch prediction service: The service simply
  runs inference on the users graph and returns the results. Keys are essential
  because there is no order guarantee on the outputs so they need to be rejoined
  to the inputs via keys or transclusion of the inputs in the outputs.
  Example:
  ```python
    def input_fn():
      features, labels = ...
      features['unique_example_id'] = ...
      features, labels
    estimator = tf.estimator.LinearClassifier(...)
    estimator = tf.contrib.estimator.forward_features(
        estimator, 'unique_example_id')
    estimator.train(...)
    assert 'unique_example_id' in estimator.predict(...)
  ```
  Args:
    estimator: A `tf.estimator.Estimator` object.
    keys: A `string`
    sparse_default_values: A dict of `str` keys mapping the name of the sparse
      features to be converted to dense, to the default value to use. Only
      sparse features indicated in the dictionary are converted to dense and the
      provided default value is used.
  Returns:
      A new `tf.estimator.Estimator` which forwards features to predictions.
  Raises:
    ValueError:
      * if `keys` is already part of `predictions`. We don't allow
        override.
      * if 'keys' does not exist in `features`.
    TypeError: if `keys` type is not one of `string` or list/tuple of `string`.
  """

  def verify_key_types(keys):  # pylint: disable=missing-docstring
    if keys is None:
      return keys
    if isinstance(keys, six.string_types):
      return [keys]
    if not isinstance(keys, (list, tuple)):
      raise TypeError('keys should be either a string or a list of strings. '
                      'Given: {}'.format(type(keys)))
    for key in keys:
      if not isinstance(key, six.string_types):
        raise TypeError('All items in the given keys list should be a string. '
                        'There exist an item with type: {}'.format(type(key)))
    return keys

  def get_keys(features):
    if keys is None:
      return features.keys()
    return keys

  def verify_keys_and_predictions(features, predictions):
    if not isinstance(predictions, dict):
      raise ValueError(
          'Predictions should be a dict to be able to forward features. '
          'Given: {}'.format(type(predictions)))
    for key in get_keys(features):
      if key not in features:
        raise ValueError(
            'keys should be exist in features. Key "{}" is not in features '
            'dict. features dict has following keys: {}. Please check '
            'arguments of forward_features.'.format(key, features.keys()))
      if key in predictions:
        raise ValueError(
            'Cannot forward feature key ({}). Since it does exist in '
            'predictions. Existing prediction keys: {}. Please check arguments '
            'of forward_features.'.format(key, predictions.keys()))

  keys = verify_key_types(keys)

  def new_model_fn(features, labels, mode, config):  # pylint: disable=missing-docstring
    spec = estimator.model_fn(features, labels, mode, config)
    predictions = spec.predictions
    if predictions is None:
      return spec
    verify_keys_and_predictions(features, predictions)
    for key in get_keys(features):
      feature = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(
          features[key])
      if sparse_default_values and (key in sparse_default_values):
        if not isinstance(feature, sparse_tensor_lib.SparseTensor):
          raise ValueError(
              'Feature ({}) is expected to be a `SparseTensor`.'.format(key))
        feature = sparse_ops.sparse_tensor_to_dense(
            feature, default_value=sparse_default_values[key])
      if not isinstance(feature, ops.Tensor):
        raise ValueError(
            'Feature ({}) should be a Tensor. Please use `keys` '
            'argument of forward_features to filter unwanted features, or'
            'add key to argument `sparse_default_values`.'
            'Type of features[{}] is {}.'.format(key, key, type(feature)))
      predictions[key] = feature
    spec = spec._replace(predictions=predictions)
    if spec.export_outputs: # CHANGES HERE
      outputs = spec.export_outputs['predict'].outputs
      outputs[key] = spec.predictions[key]
      spec.export_outputs['predict'] = tf.estimator.export.PredictOutput(outputs)
      spec.export_outputs['serving_default'] = tf.estimator.export.PredictOutput(outputs)
    return spec

  return estimator_lib.Estimator(
      model_fn=new_model_fn,
      model_dir=estimator.model_dir,
      config=estimator.config)


class ModelTrainer(object):
  """Model Trainer."""

  def __init__(self, dataset: ds.DatasetInput,
               model: base_model.BaseModel) -> None:
    self._dataset = dataset
    self._model = model
    self._estimator = model.estimator(self._model_dir())

  # TODO(ldixon): consider early stopping. Currently steps is hard coded.
  def train_with_eval(self, steps, eval_period, eval_steps):
    """
    Args:
      steps: total number of batches to train for.
      eval_period: the number of steps between evaluations.
      eval_steps: the number of batches that are evaluated per evaulation.
    """

    training_hooks = None
    if FLAGS.enable_profiling:
      training_hooks = [tf.train.ProfilerHook(save_steps=10,
          output_dir=os.path.join(self._model_dir(), 'profiler'))]

    train_spec = tf.estimator.TrainSpec(
        input_fn=self._dataset.train_input_fn,
        max_steps=steps,
        hooks=training_hooks)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=self._dataset.validate_input_fn,
        steps=eval_steps,
        throttle_secs=1,
        )
    self._estimator._config = self._estimator.config.replace(save_checkpoints_steps=eval_period)
    if FLAGS.n_export > 1:
      self._estimator._config = self._estimator.config.replace(keep_checkpoint_max=None)
    tf.estimator.train_and_evaluate(
        self._estimator,
        train_spec,
        eval_spec)

  def _model_dir(self):
    """Get Model Directory.

    Used to scope logs to a given trial (when hyper param tuning) so that they
    don't run over each other. When running locally it will just use the passed
    in model_dir.
    """
    return os.path.join(
        FLAGS.model_dir,
        json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get(
            'trial', ''))

  def _add_estimator_key(self, estimator):
    '''Adds a forward key to the model_fn of an estimator.'''
    estimator = forward_features(estimator, FLAGS.key_name)
    return estimator

  def _get_list_checkpoint(self, n_export, model_dir):
    """Get the checkpoints that we want to export.

    Args:
      n_export: Number of models to export.
      model_dir: Directory containing the checkpoints.

    Returns:
      List of checkpoint path.

    If n_export==1, we take only the last checkpoint.
    Otherwise, we consider the list of steps for each for which we have a checkpoint.
    Then we choose n_export number of checkpoints such as their steps are as equidistant as possible.  
    """

    checkpoints = file_io.get_matching_files(
        os.path.join(model_dir, 'model.ckpt-*.index'))
    checkpoints = [x.replace('.index', '') for x in checkpoints]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))

    if n_export == 1:
      return [checkpoints[-1]]

    # We want to cover a distance of (len(checkpoints) - 1): for 3 points, we have a distance of 2. 
    # with a number of points of (n_export -1): because 1 point is set at the end.
    step = float(len(checkpoints) - 1) / (n_export - 1)
    if step <= 1: # Fewer checkpoints available than the desired number.
      return checkpoints
    
    checkpoints_to_export = [checkpoints[int(i*step)] for i in range(n_export-1)]
    checkpoints_to_export.append(checkpoints[-1])
    
    return checkpoints_to_export

  def export(self, serving_input_fn):
    '''Export model as a .pb.'''
    estimator_with_key = self._add_estimator_key(self._estimator)
    
    checkpoints_to_export = self._get_list_checkpoint(FLAGS.n_export, self._model_dir())
    for checkpoint_path in checkpoints_to_export:
      version = checkpoint_path.split('-')[-1]
      estimator_with_key.export_savedmodel(
          export_dir_base=os.path.join(self._model_dir(), version),
          serving_input_receiver_fn=serving_input_fn,
          checkpoint_path=checkpoint_path)
