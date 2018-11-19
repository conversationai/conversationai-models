"""Tensorflow Estimator implementation of RNN Model with Attention"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_trainer.common import base_model
from typing import Set

FLAGS = tf.app.flags.FLAGS

# Hyperparameters
# TODO: Add validation
tf.app.flags.DEFINE_float('learning_rate', 0.00003,
                          'The learning rate to use during training.')
tf.app.flags.DEFINE_float('dropout_rate', 0.3,
                          'The dropout rate to use during training.')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning.
tf.app.flags.DEFINE_string(
    'gru_units', '128',
    'Comma delimited string for the number of hidden units in the gru layer.')
tf.app.flags.DEFINE_integer('attention_units', 64,
                            'The number of hidden units in the gru layer.')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning.
tf.app.flags.DEFINE_string(
    'dense_units', '128',
    'Comma delimited string for the number of hidden units in the dense layer.')


def attend(inputs, attention_size, attention_depth=1):
  '''Attention layer.'''

  sequence_length = tf.shape(inputs)[1] # dynamic
  final_layer_size = inputs.shape[2] # static

  x = tf.reshape(inputs, [-1, final_layer_size])
  for _ in range(attention_depth-1):
    x = tf.layers.dense(x, attention_size, activation=tf.nn.relu)
  x = tf.layers.dense(x, 1, activation=None)
  logits = tf.reshape(x, [-1, sequence_length, 1])
  alphas = tf.nn.softmax(logits, dim=1)

  output = tf.reduce_sum(inputs * alphas, 1)

  return output, alphas


class TFRNNModel(base_model.BaseModel):

  def __init__(self, 
    target_labels: Set[str]) -> None:
    self._target_labels = target_labels

  @staticmethod
  def hparams():
    gru_units = [int(units) for units in FLAGS.gru_units.split(',')]
    dense_units = [int(units) for units in FLAGS.dense_units.split(',')]
    hparams = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        dropout_rate=FLAGS.dropout_rate,
        gru_units=gru_units,
        attention_units=FLAGS.attention_units,
        dense_units=dense_units)
    return hparams

  def estimator(self, model_dir):
    estimator = tf.estimator.Estimator(
        model_fn=self._model_fn,
        params=self.hparams(),
        config=tf.estimator.RunConfig(model_dir=model_dir))
    return estimator

  def _model_fn(self, features, labels, mode, params, config):
    inputs = features[base_model.TOKENS_FEATURE_KEY]
    batch_size = tf.shape(inputs)[0]

    rnn_layers = [
        tf.nn.rnn_cell.GRUCell(num_units=size, activation=tf.nn.tanh)
        for size in params.gru_units
    ]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # TODO: make bidirectional
    outputs, states = tf.nn.dynamic_rnn(
        multi_rnn_cell,
        inputs,
        dtype=tf.float32)

    # TODO: Handle sequence length in the attention layer (via a mask).
    #       Padded elements should not be part of the average.
    logits, _ = attend(
        inputs=outputs,
        attention_size=params.attention_units)

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
