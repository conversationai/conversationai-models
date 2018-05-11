"""RNN"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, GRU, Dense, Embedding, Dropout, Bidirectional, TimeDistributed, Flatten, Dot
from keras.models import Model
from keras_trainer import base_model
from keras_trainer.custom_metrics import auc_roc


class RNNModel(base_model.BaseModel):
  """ RNN

  hparams:
    embedding_dim
    vocab_size
    train_embedding
  """

  def __init__(self, embeddings_matrix, hparams):
    self.embeddings_matrix = embeddings_matrix
    self.hparams = hparams

  def get_model(self):
    sequence_length = self.hparams.sequence_length

    I = Input(shape=(sequence_length,), dtype='float32')
    E = Embedding(
        self.hparams.vocab_size,
        self.hparams.embedding_dim,
        weights=[self.embeddings_matrix],
        input_length=sequence_length,
        trainable=self.hparams.train_embedding)(
            I)
    H = Bidirectional(GRU(128, return_sequences=True))(E)
    A = TimeDistributed(Dense(3), input_shape=(sequence_length, 256))(H)
    A = Flatten()(A)
    A = Dense(sequence_length, activation='softmax')(A)
    X = Dot((1, 1))([H, A])
    X = Dense(128, activation='relu')(X)
    X = Dropout(self.hparams.dropout_rate)(X)
    Output = Dense(6, activation='sigmoid')(X)

    model = Model(inputs=I, outputs=Output)
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy', auc_roc])

    print(model.summary())
    return model
