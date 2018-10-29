"""Tensorflow Estimator implementation of RNN Model with Attention"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
from tf_trainer.common import base_model
from typing import Set

FLAGS = tf.app.flags.FLAGS

# Hyperparameters
# TODO: Add validation
tf.app.flags.DEFINE_float('learning_rate', 0.00003,
                          'The learning rate to use during training.')
tf.app.flags.DEFINE_float('dropout_rate', 0.3,
                          'The dropout rate to use during training.')
tf.app.flags.DEFINE_string(
    'model_spec',
    'https://tfhub.dev/google/universal-sentence-encoder/2',
    'The url of the TF Hub sentence encoding module to use.')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning.
tf.app.flags.DEFINE_string(
    'dense_units', '512,128,64',
    'Comma delimited string for the number of hidden units in the dense layer.')


class TFRNNModel(base_model.BaseModel):

  def __init__(self, 
    text_feature_name: str, 
    target_labels: Set[str]) -> None:
    self._text_feature_name = text_feature_name
    self._target_labels = target_labels

  @staticmethod
  def hparams():
    dense_units = [int(units) for units in FLAGS.dense_units.split(',')]
    hparams = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        dropout_rate=FLAGS.dropout_rate,
        dense_units=dense_units)
    return hparams

  def estimator(self, model_dir):
    estimator = tf.estimator.Estimator(
        model_fn=self._model_fn,
        params=self.hparams(),
        config=tf.estimator.RunConfig(model_dir=model_dir))
    return estimator

  def _model_fn(self, features, labels, mode, params, config):
    embedded_text_feature_column = hub.text_embedding_column(
      key=self._text_feature_name, module_spec=FLAGS.model_spec)
    inputs = tf.feature_column.input_layer(
        features, [embedded_text_feature_column])

    batch_size = tf.shape(inputs)[0]

    logits = inputs
    for num_units in params.dense_units:
      logits = tf.layers.dense(
          inputs=logits, units=num_units, activation=tf.nn.relu)
      logits = tf.layers.dropout(logits, rate=params.dropout_rate)
    logits = tf.layers.dense(
        inputs=logits, units=len(self._target_labels), activation=None)

    output_heads = [
        tf.contrib.estimator.binary_classification_head(name=name)
        for name in self._target_labels
    ]
    multihead = tf.contrib.estimator.multi_head(output_heads)

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    return multihead.create_estimator_spec(
        features=features,
        labels=labels,
        mode=mode,
        logits=logits,
        optimizer=optimizer)
