"""RNN"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, GRU, Dense, Embedding, Dropout, Bidirectional
from keras.models import Model
from keras_trainer import base_model
from keras_trainer.custom_metrics import auc_roc


class RNN(base_model.BaseModel):
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
    I = Input(shape=(self.hparams.sequence_length,), dtype='float32')
    E = Embedding(
        self.hparams.vocab_size,
        self.hparams.embedding_dim,
        weights=[self.embeddings_matrix],
        input_length=self.hparams.sequence_length,
        trainable=self.hparams.train_embedding)(I)
    X = Bidirectional(GRU(128, return_sequences=False))(E)
    X = Dense(128, activation='relu')(X)
    X = Dropout(self.hparams.dropout_rate)(X)
    Output = Dense(6, activation='sigmoid')(X)

    model = Model(inputs=I, outputs=Output)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', auc_roc])

    print(model.summary())
    return model

