"""Keras CNN Model"""

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
# TODO: add link to relevant public issue/bug/documentation?
tf.app.flags.DEFINE_string(
    'filter_sizes', '5',
    'Comma delimited string for the sizes of convolution filters.')
tf.app.flags.DEFINE_integer(
    'num_filters', 128,
    'Number of convolutional filters for every convolutional layer.')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning.
# TODO: add link to relevant public issue/bug/documentation?
tf.app.flags.DEFINE_string(
    'dense_units', '128',
    'Comma delimited string for the number of hidden units in the dense layer.')


class KerasCNNModel(base_keras_model.BaseKerasModel):
  """Keras CNN Model

  Keras implementation of a CNN. Inputs should be
  sequences of word embeddings.
  """

  MAX_SEQUENCE_LENGTH = 300

  def __init__(self, labels: List[str], optimizer='adam') -> None:
    self._labels = labels

  def hparams(self):
    filter_sizes = [int(units) for units in FLAGS.filter_sizes.split(',')]
    dense_units = [int(units) for units in FLAGS.dense_units.split(',')]
    return tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        dropout_rate=FLAGS.dropout_rate,
        filter_sizes=filter_sizes,
        num_filters=FLAGS.num_filters,
        dense_units=dense_units)

  # Local function you are expected to overwrite.
  def _get_keras_model(self) -> models.Model:
    I = layers.Input(
        shape=(KerasCNNModel.MAX_SEQUENCE_LENGTH, 300),
        dtype='float32',
        name=base_model.TOKENS_FEATURE_KEY)

    # Convolutional Layers
    X = I
    for filter_size in self.hparams().filter_sizes:
        X = layers.Conv1D(self.hparams().num_filters, filter_size, activation='relu', padding='same')(X)
    X = layers.GlobalAveragePooling1D()(X)

    # Dense
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
