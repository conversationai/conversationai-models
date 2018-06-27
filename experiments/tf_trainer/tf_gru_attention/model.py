"""Tensorflow Estimator implementation of RNN Model with Attention"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_trainer.common import base_model
from typing import List

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


class TFRNNModel(base_model.BaseModel):

  def __init__(self, text_feature_name: str, target_labels: List[str]) -> None:
    self._text_feature_name = text_feature_name
    self._target_labels = target_labels

  @staticmethod
  def hparams():
    gru_units = [int(units) for units in FLAGS.gru_units.split(',')]
    dense_units = [int(units) for units in FLAGS.dense_units.split(',')]
    hparams = tf.contrib.training.HParams(
        max_seq_length=300,
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
    inputs = features[self._text_feature_name]
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
        sequence_length=tf.fill(dims=[batch_size], value=params.max_seq_length),
        dtype=tf.float32)

    # TF needs help understanding sequence length (I think because we're using
    # dynamic_rnn)
    outputs = tf.reshape(
        outputs, [batch_size, params.max_seq_length, params.gru_units[-1]])

    unstacked_outputs = tf.unstack(outputs, num=params.max_seq_length, axis=1)

    attention = tf.expand_dims(
        tf.nn.softmax(
            tf.concat(
                [
                    tf.layers.dense(
                        inputs=tf.layers.dense(
                            inputs=output,
                            units=params.attention_units,
                            activation=tf.nn.relu),
                        units=1,
                        activation=None) for output in unstacked_outputs
                ],
                axis=1),
            axis=-1), -1)

    weighted_output = tf.multiply(attention, outputs)
    weighted_output = tf.reduce_sum(weighted_output, -2)

    logits = weighted_output
    for num_units in params.dense_units:
      logits = tf.layers.dense(
          inputs=weighted_output, units=num_units, activation=tf.nn.relu)
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
