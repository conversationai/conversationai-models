"""Keras RNN Model with Attention"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tf_trainer.common import base_model
from tf_trainer.common import base_keras_model
import tensorflow as tf

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


class KerasRNNModel(base_keras_model.BaseKerasModel):
  """Keras RNN Model

  Keras implementation of a bidirectional GRU with attention. Inputs should be
  sequences of word embeddings.
  """

  def __init__(self,
               labels: List[str],
               embedding_size: int,
               optimizer='adam') -> None:
    self._labels = labels
    self._embedding_size = embedding_size

  def hparams(self):
    gru_units = [int(units) for units in FLAGS.gru_units.split(',')]
    dense_units = [int(units) for units in FLAGS.dense_units.split(',')]
    return tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        dropout_rate=FLAGS.dropout_rate,
        gru_units=gru_units,
        attention_units=FLAGS.attention_units,
        dense_units=dense_units)

  def _get_keras_model(self) -> models.Model:
    I = layers.Input(
        shape=(None, self._embedding_size),
        dtype='float32',
        name=base_model.TOKENS_FEATURE_KEY)

    # Bidirectional GRU
    H = I
    for num_units in self.hparams().gru_units:
      H = layers.Bidirectional(layers.GRU(num_units, return_sequences=True))(I)

    # Attention
    last_gru_units = self.hparams().gru_units[-1] * 2  # x2 because bidirectional
    A = layers.TimeDistributed(
        layers.Dense(self.hparams().attention_units, activation='relu'),
        input_shape=(None, last_gru_units))(
            H)
    A = layers.TimeDistributed(layers.Dense(1))(A)
    A = layers.Flatten()(A)
    A = layers.Activation('softmax')(A)

    # Dense
    X = layers.Dot((1, 1))([H, A])
    X = layers.Flatten()(X)
    for num_units in self.hparams().dense_units:
      X = layers.Dense(num_units, activation='relu')(X)
      X = layers.Dropout(self.hparams().dropout_rate)(X)

    # Outputs
    outputs = []
    for label in self._labels:
      outputs.append(layers.Dense(1, activation='sigmoid', name=label)(X))

    model = models.Model(inputs=I, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(lr=self.hparams().learning_rate),
        loss='binary_crossentropy',
        metrics=['binary_accuracy', super().roc_auc])

    tf.logging.info(model.summary())
    return model
