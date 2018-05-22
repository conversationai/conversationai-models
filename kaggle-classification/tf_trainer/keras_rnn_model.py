"""Keras RNN Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tf_trainer import custom_metrics


class KerasRNNModel():
  """Keras RNN Model

  Keras implementation of a bidirectional GRU with attention. Inputs should be
  sequences of word embeddings.
  """

  def __init__(self, labels):
    self._labels = labels

  def get_model(self):
    sequence_length = 300
    I = layers.Input(
        shape=(sequence_length, 100), dtype='float32', name='comment_text')
    H = layers.Bidirectional(layers.GRU(128, return_sequences=True))(I)
    A = layers.TimeDistributed(
        layers.Dense(128, activation='relu'),
        input_shape=(sequence_length, 256))(
            H)
    A = layers.TimeDistributed(layers.Dense(1, activation='softmax'))(H)
    X = layers.Dot((1, 1))([H, A])
    X = layers.Flatten()(X)
    X = layers.Dense(128, activation='relu')(X)
    X = layers.Dropout(0.3)(X)
    outputs = []
    for label in self._labels:
      outputs.append(layers.Dense(1, activation='sigmoid', name=label)(X))

    model = models.Model(inputs=I, outputs=outputs)
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy', custom_metrics.roc_auc])

    print(model.summary())
    return model
