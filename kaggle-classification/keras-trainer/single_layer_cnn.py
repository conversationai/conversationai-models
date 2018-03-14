"""Model class for a single layer CNN"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base_model import BaseModel
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras.layers import Activation
from keras.layers import Concatenate
from keras.models import Model


class SingleLayerCnn(BaseModel):
  """Single Layer Based CNN"""

  def __init__(self, embeddings_matrix, embedding_dim, vocab_size, sequence_length, dropout_rate):
    self.embeddings_matrix = embeddings_matrix
    self.embedding_dim = embedding_dim
    self.vocab_size = vocab_size
    self.sequence_length = sequence_length
    self.dropout_rate = dropout_rate

  def get_model(self):
    I = Input(shape=(self.sequence_length,), dtype='float32')
    E = Embedding(
        self.vocab_size,
        self.embedding_dim,
        weights=[self.embeddings_matrix],
        input_length=self.sequence_length,
        trainable=False)(I)
    X5 = Conv1D(128, 5, activation='relu', padding='same')(E)
    X5 = MaxPooling1D(self.sequence_length, padding='same')(X5)
    X4 = Conv1D(128, 4, activation='relu', padding='same')(E)
    X4 = MaxPooling1D(self.sequence_length, padding='same')(X4)
    X3 = Conv1D(128, 3, activation='relu', padding='same')(E)
    X3 = MaxPooling1D(self.sequence_length, padding='same')(X3)
    X = Concatenate(axis=-1)([X5, X4, X3])
    X = Flatten()(X)
    X = Dropout(self.dropout_rate)(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(self.dropout_rate)(X)
    toxic_out = Dense(1, activation='sigmoid', name='toxic')(X)
    severe_toxic_out = Dense(1, activation='sigmoid', name='severe_toxic')(X)
    obscene_out = Dense(1, activation='sigmoid', name='obscene')(X)
    threat_out = Dense(1, activation='sigmoid', name='threat')(X)
    insult_out = Dense(1, activation='sigmoid', name='insult')(X)
    identity_hate_out = Dense(1, activation='sigmoid', name='identity_hate')(X)

    model = Model(inputs=I, outputs=[toxic_out, severe_toxic_out, obscene_out, threat_out, insult_out, identity_hate_out])
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model
