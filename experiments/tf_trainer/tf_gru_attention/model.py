"""Tensorflow Estimator implementation of RNN Model with Attention"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class TFRNNModel():
  DEFAULT_HPARAMS = tf.contrib.training.HParams(max_seq_length=300)

  def __init__(self, text_feature_name, target_labels):
    self._text_feature_name = text_feature_name
    self._target_labels = target_labels

  def estimator(self, run_config):
    estimator = tf.estimator.Estimator(
        model_fn=self._model_fn,
        params=TFRNNModel.DEFAULT_HPARAMS,
        config=run_config)
    return estimator

  def _model_fn(self, features, labels, mode, params, config):
    inputs = features[self._text_feature_name]
    batch_size = tf.shape(inputs)[0]

    rnn_layers = [
        tf.nn.rnn_cell.GRUCell(num_units=size, activation=tf.nn.tanh)
        for size in [128]
    ]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    outputs, states = tf.nn.dynamic_rnn(
        multi_rnn_cell,
        inputs,
        sequence_length=tf.fill(dims=[batch_size], value=params.max_seq_length),
        dtype=tf.float32)

    # TF needs help understanding sequence length (I think because we're using
    # dynamic_rnn)
    outputs = tf.reshape(outputs, [batch_size, params.max_seq_length, 128])

    unstacked_outputs = tf.unstack(outputs, num=params.max_seq_length, axis=1)

    attention = tf.expand_dims(
        tf.nn.softmax(
            tf.concat(
                [
                    tf.layers.dense(
                        inputs=tf.layers.dense(
                            inputs=output, units=64, activation=tf.nn.relu),
                        units=1,
                        activation=None) for output in unstacked_outputs
                ],
                axis=1),
            axis=-1), -1)

    weighted_output = tf.multiply(attention, outputs)
    weighted_output = tf.reduce_sum(weighted_output, -2)

    logits = tf.layers.dense(
        inputs=weighted_output, units=128, activation=tf.nn.relu)
    logits = tf.layers.dropout(logits, rate=0.3)
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
      optimizer = tf.train.AdamOptimizer(learning_rate=0.00005)

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
          'auroc': tf.metrics.auc(labels, probabilities)
      }

      # Provide an estimator spec for `ModeKeys.EVAL` modes.
      return tf.estimator.EstimatorSpec(
          mode, loss=loss, eval_metric_ops=eval_metric_ops)
