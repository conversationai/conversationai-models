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

import comet_ml
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tf_trainer.common import base_model
from tf_trainer.common import dataset_input as ds


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_path', None,
                           'Path to the training data TFRecord file.')
tf.app.flags.DEFINE_string('validate_path', None,
                           'Path to the validation data TFRecord file.')
tf.app.flags.DEFINE_string('model_dir', None,
                           "Directory for the Estimator's model directory.")
tf.app.flags.DEFINE_string('comet_key_file', None,
                           'Path to file containing comet.ml api key.')
tf.app.flags.DEFINE_string('comet_team_name', None,
                           'Name of comet team that tracks results.')
tf.app.flags.DEFINE_string('comet_project_name', None,
                           'Name of comet project that tracks results.')
tf.app.flags.DEFINE_bool('enable_profiling', False,
                           'Enable profiler hook in estimator.')

tf.app.flags.mark_flag_as_required('train_path')
tf.app.flags.mark_flag_as_required('validate_path')
tf.app.flags.mark_flag_as_required('model_dir')






import six

from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.export.export_output import PredictOutput
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.training import optimizer as optimizer_lib




def forward_featuresFPROST(estimator, keys=None, sparse_default_values=None):
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
    keys: A `string` or a `list` of `string`. If it is `None`, all of the
      `features` in `dict` is forwarded to the `predictions`. If it is a
      `string`, only given key is forwarded. If it is a `list` of strings, all
      the given `keys` are forwarded.
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
    print ('EXPORT_OUTPUTS')
    print (spec.export_outputs)
    if spec.export_outputs:
      for ekey in ['predict', 'serving_default']:
        print('ekey: ' + ekey)
        print('t1: ' + str(ekey in spec.export_outputs))
        print('t2: ' + str(isinstance(spec.export_outputs[ekey], PredictOutput)))
        print ('export_output: ')
        print (spec.export_outputs[ekey])
        # if (ekey in spec.export_outputs and
        #     isinstance(spec.export_outputs[ekey],
        #                PredictOutput)):
        if (ekey in spec.export_outputs):
          export_outputs = spec.export_outputs[ekey].outputs
          for key in get_keys(features):
            print ('key: '+ key)
            export_outputs[key] = predictions[key]

    return spec

  return estimator_lib.Estimator(
      model_fn=new_model_fn,
      model_dir=estimator.model_dir,
      config=estimator.config)



def forward_key_to_export(estimator):
  """Forwards record key to output during inference.

  Temporary workaround. The key and its value will be extracted from input
  tensors and returned in the prediction dictionary. This is useful to pass
  record key identifiers. Code came from:
  https://towardsdatascience.com/how-to-extend-a-canned-tensorflow-estimator-to-add-more-evaluation-metrics-and-to-pass-through-ddf66cd3047d
  This shouldn't be necessary. (CL/187793590 was filed to update extenders.py
  with this code).

  Args:
      estimator: `Estimator` being modified.

  Returns:
      A modified `Estimator`
  """
  config = estimator.config
  def model_fn2(features, labels, mode):
    estimator_spec = estimator._call_model_fn(
        features, labels, mode, config=config)
    import six
    from tensorflow.python.estimator.canned import head as head_lib
    from tensorflow.python.estimator.canned import metric_keys
    from tensorflow.python.estimator.export import export_output as export_output_lib

    if mode==tf.estimator.ModeKeys.PREDICT:
      if estimator_spec.export_outputs:
        outputs = estimator_spec.export_outputs['predict'].outputs
        outputs['comment_key'] = estimator_spec.predictions['comment_key']
        estimator_spec.export_outputs['predict'] = tf.estimator.export.PredictOutput(outputs)
        estimator_spec.export_outputs['serving_default'] = tf.estimator.export.PredictOutput(outputs)
    return estimator_spec
  return tf.estimator.Estimator(model_fn=model_fn2, config=config)


class ModelTrainer(object):
  """Model Trainer.

  Convenient way to run a text classification estimator, supporting comet.ml
  outputs.
  """

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
    experiment = None
    if FLAGS.comet_key_file is not None:
      experiment = self._setup_comet()
    num_itr = int(steps / eval_period)

    writer = tf.summary.FileWriter(self._model_dir())
    for i in range(num_itr):
      hooks = None
      if FLAGS.enable_profiling:
        hooks = [tf.train.ProfilerHook(save_steps=10,
                                       output_dir=os.path.join(self._model_dir(), 'profiler'))]
      self._estimator.train(
          input_fn=self._dataset.train_input_fn,
          steps=eval_period,
          hooks=hooks)
      metrics = self._estimator.evaluate(
          input_fn=self._dataset.validate_input_fn, steps=eval_steps)
      if experiment is not None:
        tf.logging.info('Logging metrics to comet.ml: {}'.format(metrics))
        experiment.log_multiple_metrics(metrics)
      predictions = self._estimator.predict(self._dataset.bias_input_fn)
      proba_list = []
      import numpy as np
      for prediction in predictions:
        proba_toxic = prediction[('frac_neg', 'probabilities')][1]
        proba_list.append(proba_toxic)
      summary = tf.Summary(value=[tf.Summary.Value(tag='unintended_bias',
                                                   simple_value=np.mean(proba_list))])
      writer.add_summary(summary, i * eval_period)
      tf.logging.info(metrics)

  def _setup_comet(self):
    with tf.gfile.GFile(FLAGS.comet_key_file) as key_file:
      key = key_file.read().rstrip()
    experiment = comet_ml.Experiment(
        api_key=key,
        project_name=FLAGS.comet_project_name,
        team_name=FLAGS.comet_team_name,
        auto_param_logging=False,
        parse_args=False)
    experiment.log_parameter('train_path', FLAGS.train_path)
    experiment.log_parameter('validate_path', FLAGS.validate_path)
    experiment.log_parameter('model_dir', self._model_dir())
    experiment.log_multiple_params(self._model.hparams().values())
    return experiment

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
    estimator = forward_featuresFPROST(estimator, FLAGS.key_name) #tf.contrib.estimator.forward_features(estimator, FLAGS.key_name)
    # estimator = forward_key_to_export(estimator)
    return estimator

  def export(self, serving_input_fn):
    '''Export model as a .pb.'''
    estimator_with_key = self._add_estimator_key(self._estimator)
    estimator_with_key.export_savedmodel(
        export_dir_base=self._model_dir(),
        serving_input_receiver_fn=serving_input_fn)
    logging.info('Model exported at {}'.format(self._model_dir()))
