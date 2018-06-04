"""Keras RNN Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tf_trainer import base_keras_model
import tensorflow as tf

from typing import Set


class KerasRNNModel(base_keras_model.BaseKerasModel):
  """Keras RNN Model

  Keras implementation of a bidirectional GRU with attention. Inputs should be
  sequences of word embeddings.
  """

  MAX_SEQUENCE_LENGTH = 300

  def __init__(self, labels: Set[str], optimizer='adam') -> None:
    self._labels = labels

  def _get_keras_model(self) -> models.Model:
    I = layers.Input(
        shape=(KerasRNNModel.MAX_SEQUENCE_LENGTH, 100),
        dtype='float32',
        name='comment_text')
    H = layers.Bidirectional(layers.GRU(128, return_sequences=True))(I)
    A = layers.TimeDistributed(
        layers.Dense(64, activation='relu'),
        input_shape=(KerasRNNModel.MAX_SEQUENCE_LENGTH, 256))(
            H)
    A = layers.TimeDistributed(layers.Dense(1))(A)
    A = layers.Flatten()(A)
    A = layers.Activation('softmax')(A)
    X = layers.Dot((1, 1))([H, A])
    X = layers.Flatten()(X)
    X = layers.Dense(128, activation='relu')(X)
    X = layers.Dropout(0.3)(X)
    outputs = []
    for label in self._labels:
      outputs.append(layers.Dense(1, activation='sigmoid', name=label)(X))

    model = models.Model(inputs=I, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(lr=0.000003),
        loss='binary_crossentropy',
        metrics=['accuracy', super().roc_auc])

    tf.logging.info(model.summary())
    return model
