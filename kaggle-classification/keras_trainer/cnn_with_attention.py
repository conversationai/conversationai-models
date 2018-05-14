"""Model class for a single layer CNN"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import AveragePooling1D
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Multiply
from keras.models import Model
from keras.layers import Permute
from keras_trainer import base_model
from keras.layers import Activation
from keras_trainer.custom_metrics import auc_roc


class CNNWithAttention(base_model.BaseModel):
  """Single Layer Based CNN

  hparams:
    embedding_dim
    vocab_size
    sequence_length
    dropout_rate
    train_embedding
  """

  def __init__(self, embeddings_matrix, hparams, labels):
    self.embeddings_matrix = embeddings_matrix
    self.hparams = hparams
    self.labels = labels
    self.num_labels = len(labels)

  def get_model(self):
    I = Input(shape=(self.hparams.sequence_length,), dtype='float32')
    E = Embedding(
        self.hparams.vocab_size,
        self.hparams.embedding_dim,
        weights=[self.embeddings_matrix],
        input_length=self.hparams.sequence_length,
        trainable=self.hparams.train_embedding)(I)
    C = []
    A = []
    P = []
    for i, size in enumerate(self.hparams.filter_sizes):
        C.append(Conv1D(self.hparams.num_filters[i], size, activation='relu', padding='same')(E))
        A.append(Dense(self.hparams.attention_intermediate_size, activation = 'relu')(C[i]))
        A[i] = Dense(1, use_bias=False)(A[i])
        # Permute trick to apply softmax to second to last layer.
        A[i] = Permute((2,1))(A[i])
        A[i] = Activation('softmax')(A[i])
        A[i] = Permute((2,1))(A[i])
        P.append(Multiply()([A[i], C[i]]))
        P[i] = AveragePooling1D(self.hparams.sequence_length, padding='same')(P[i])
    X = Concatenate(axis=-1)(P)
    X = Flatten()(X)
    X = Dropout(self.hparams.dropout_rate)(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(self.hparams.dropout_rate)(X)
    Output = Dense(self.num_labels, activation='sigmoid', name='outputs')(X)

    model = Model(inputs=I, outputs=Output)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', auc_roc])
    print(model.summary())
    return model
