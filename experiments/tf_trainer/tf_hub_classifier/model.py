"""Tensorflow Estimator using TF Hub universal sentence encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
from tf_trainer.common import base_model
from typing import List

FLAGS = tf.app.flags.FLAGS

# Hyperparameters
# TODO: Add validation
tf.app.flags.DEFINE_float('learning_rate', 0.00003,
                          'The learning rate to use during training.')
tf.app.flags.DEFINE_float('dropout_rate', 0.15,
                          'The dropout rate to use during training.')
tf.app.flags.DEFINE_string(
    'model_spec',
    'https://tfhub.dev/google/universal-sentence-encoder-large/3'
    'The url of the TF Hub sentence encoding module to use.')
tf.app.flags.DEFINE_bool('trainable', True,
                         'What to pass for the TF Hub trainable parameter.')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning. The length of the list
# determines the number of layers, and the size of each layer.
tf.app.flags.DEFINE_string(
    'dense_units', '1024,1024,512',
    'Comma delimited string for the number of hidden units in the dense layers.'
)


class TFHubClassifierModel(base_model.BaseModel):

  def __init__(self, target_labels: List[str]) -> None:
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
        key=base_model.TEXT_FEATURE_KEY,
        module_spec=FLAGS.model_spec,
        trainable=FLAGS.trainable)
    inputs = tf.feature_column.input_layer(features,
                                           [embedded_text_feature_column])

    batch_size = tf.shape(inputs)[0]

    logits = inputs
    for num_units in params.dense_units:
      logits = tf.layers.dense(
          inputs=logits, units=num_units, activation=tf.nn.relu)
      logits = tf.layers.dropout(logits, rate=params.dropout_rate)
    logits = tf.layers.dense(
        inputs=logits, units=len(self._target_labels), activation=None)

    output_heads = [
        tf.contrib.estimator.binary_classification_head(
            name=name, weight_column=name + '_weight')
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
