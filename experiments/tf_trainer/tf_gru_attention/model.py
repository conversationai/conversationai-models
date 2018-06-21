"""Tensorflow Estimator implementation of RNN Model with Attention"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

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


class TFRNNModel():

  def __init__(self, text_feature_name, target_labels):
    self._text_feature_name = text_feature_name
    self._target_labels = target_labels

    gru_units = [int(units) for units in FLAGS.gru_units.split(',')]
    dense_units = [int(units) for units in FLAGS.dense_units.split(',')]
    self._hparams = tf.contrib.training.HParams(
        max_seq_length=300,
        learning_rate=FLAGS.learning_rate,
        dropout_rate=FLAGS.dropout_rate,
        gru_units=gru_units,
        attention_units=FLAGS.attention_units,
        dense_units=dense_units)

  @property
  def hparams(self):
    return self._hparams

  def estimator(self, run_config):
    estimator = tf.estimator.Estimator(
        model_fn=self._model_fn, params=self.hparams, config=run_config)
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
    outputs = tf.reshape(outputs,
                         [batch_size, params.max_seq_length, gru_units[-1]])

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

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
      probabilities = tf.nn.sigmoid(logits)
      guesses = tf.to_int32(probabilities > 0.5)

      # Convert predicted_indices back into strings
      predictions = {
          'class': tf.gather(self._target_labels, guesses),
          'probabilities': probabilities
      }
      export_outputs = {
          'prediction': tf.estimator.export.PredictOutput(predictions)
      }

      # Provide an estimator spec for `ModeKeys.PREDICT` modes.
      return tf.estimator.EstimatorSpec(
          mode, predictions=predictions, export_outputs=export_outputs)

    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
      # Create Optimiser
      optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

      # Create training operation
      train_op = optimizer.minimize(
          loss=loss, global_step=tf.train.get_global_step())

      # Provide an estimator spec for `ModeKeys.TRAIN` modes.
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
      probabilities = tf.sigmoid(logits)
      guesses = tf.to_int32(probabilities > 0.5)

      eval_metric_ops = {
          'accuracy': tf.metrics.accuracy(labels, guesses),
          'roc_auc': tf.metrics.auc(labels, probabilities)
      }

      # Provide an estimator spec for `ModeKeys.EVAL` modes.
      return tf.estimator.EstimatorSpec(
          mode, loss=loss, eval_metric_ops=eval_metric_ops)
