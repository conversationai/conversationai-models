"""Model Runner class for text classification"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import comet_ml
import abc
from typing import Dict, Any

from tf_trainer.common import dataset_input as ds
from tf_trainer.common import tfrecord_input
from tf_trainer.common import text_preprocessor
from tf_trainer.common import types

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_path', None,
                           'Path to the training data TFRecord file.')
tf.app.flags.DEFINE_string('validate_path', None,
                           'Path to the validation data TFRecord file.')
tf.app.flags.DEFINE_string('model_dir', None,
                           "Directory for the Estimator's model directory.")
tf.app.flags.DEFINE_string('comet_key', None, 'Your comet.ml api key.')
tf.app.flags.DEFINE_string('comet_team_name', None,
                           'Name of comet team that tracks results.')
tf.app.flags.DEFINE_string('comet_project_name', None,
                           'Name of comet project that tracks results.')

tf.app.flags.mark_flag_as_required('train_path')
tf.app.flags.mark_flag_as_required('validate_path')
tf.app.flags.mark_flag_as_required('model_dir')


class ModelRunner(abc.ABC):
  """Text Classification Model Runner.

  Convenient way to run a text classification estimator.
  """

  def __init__(self) -> None:
    pass

  @abc.abstractmethod
  def dataset_input(self, train_path: str,
                    validate_path: str) -> ds.DatasetInput:
    pass

  @abc.abstractmethod
  def estimator(self, model_dir: str) -> tf.estimator.Estimator:
    pass

  def log_params(self) -> Dict[str, Any]:
    """Override to log more params"""
    return {}

  def train_with_eval(self, steps, eval_period, eval_steps):
    if FLAGS.comet_key:
      experiment = self._setup_comet()
    num_itr = int(steps / eval_period)
    dataset = self.dataset_input(FLAGS.train_path, FLAGS.validate_path)
    estimator = self.estimator(FLAGS.model_dir)

    for _ in range(num_itr):
      estimator.train(input_fn=dataset.train_input_fn, steps=eval_period)
      metrics = estimator.evaluate(
          input_fn=dataset.validate_input_fn, steps=eval_steps)
      if experiment:
        experiment.log_multiple_metrics(metrics)
      tf.logging.info(metrics)

  def _setup_comet(self):
    experiment = comet_ml.Experiment(
        api_key=FLAGS.comet_key,
        project_name=FLAGS.comet_project_name,
        team_name=FLAGS.comet_team_name,
        auto_param_logging=False,
        parse_args=False)
    experiment.log_parameter('train_path', FLAGS.train_path)
    experiment.log_parameter('validate_path', FLAGS.validate_path)
    experiment.log_parameter('model_dir', FLAGS.model_dir)
    experiment.log_multiple_params(self.log_params())
    return experiment
