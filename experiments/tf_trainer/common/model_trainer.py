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
tf.app.flags.DEFINE_bool('enable_profiling', True,
                           'Enable profiler hook in estimator.')

tf.app.flags.mark_flag_as_required('train_path')
tf.app.flags.mark_flag_as_required('validate_path')
tf.app.flags.mark_flag_as_required('model_dir')


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
    if estimator_spec.export_outputs:
      for ekey in ['predict', 'serving_default']:
        estimator_spec.export_outputs[ekey] = \
            tf.estimator.export.PredictOutput(estimator_spec.predictions)
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

    for _ in range(num_itr):
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
    estimator = tf.contrib.estimator.forward_features(estimator, FLAGS.key_name)
    estimator = forward_key_to_export(estimator)
    return estimator

  def export(self, serving_input_fn):
    '''Export model as a .pb.'''
    estimator_with_key = self._add_estimator_key(self._estimator)
    estimator_with_key.export_savedmodel(
        export_dir_base=self._model_dir(),
        serving_input_receiver_fn=serving_input_fn)
    logging.info('Model exported at {}'.format(self._model_dir()))
