"""
Classifiers for the Toxic Comment Classification Kaggle challenge,
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

To run locally:
  python keras-trainer/model.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras.layers import Activation
from keras.layers import Concatenate
from keras.models import load_model
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from os.path import expanduser
from sklearn import metrics

FLAGS = None

DEFAULT_HPARAMS = tf.contrib.training.HParams(
    learning_rate=0.00005,
    dropout_rate=0.3,
    batch_size=128,
    epochs=5,
    max_sequence_length=250,
    embedding_dim=100)

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


class AttentionToxModel():
  """Toxicity model using CNN + Attention"""
  
  def __init__(self,
               model_path,
               embeddings_path,
               hparams=DEFAULT_HPARAMS):
    self.model_path = model_path
    self.embeddings_path = embeddings_path
    self.hparams = hparams
    print('Setting up tokenizer...')
    self.tokenizer = self._setup_tokenizer()
    print('Setting up embedding matrix...')
    self.embedding_matrix = self._setup_embedding_matrix()
    print('Loading model...')
    self._load_model()
  
  def train(self, train):
    model = self._build_model()
    train_comment = self._prep_texts(train['comment_text'])
    train_labels = [train[label] for label in LABELS]

    callbacks = [
        ModelCheckpoint(
            self.model_path, save_best_only=True, verbose=True),
        EarlyStopping(
            monitor='val_loss', mode='auto')
    ]

    model.fit(
        x=train_comment, y=train_labels,
        batch_size=self.hparams.batch_size,
        epochs=self.hparams.epochs,
        validation_split=0.1,
        callbacks=callbacks)
    
    self._load_model()

  def predict(self, texts):
    data = self._prep_texts(texts)
    return self.model.predict(data)

  def score_auc(self, data):
    predictions = self.predict(data['comment_text'])
    scores = []
    for idx, label in enumerate(LABELS):
      labels = np.array(data['toxic'])
      score = metrics.roc_auc_score(labels, predictions[idx].flatten())
      scores.append(score)
      print('{} has AUC {}'.format(label, score))
    print('Avg AUC {}'.format(np.mean(scores)))

  def _prep_texts(self, texts):
    return pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=self.hparams.max_sequence_length)

  def _load_model(self):
    try:
      self.model = load_model(self.model_path)
      print('Model loaded from: {}'.format(self.model_path))
    except IOError:
      print('Could not load model at: {}'.format(self.model_path))

  def _setup_tokenizer(self):
    words = []
    with open(self.embeddings_path) as f:
      for line in f:
        words.append(line.split()[0])
    # TODO: configure OOV token
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words)
    return tokenizer

  def _setup_embedding_matrix(self):
    embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1, self.hparams.embedding_dim))
    with open(self.embeddings_path) as f:
      for line in f:
        values = line.split()
        word = values[0]
        if word in self.tokenizer.word_index:
          word_idx = self.tokenizer.word_index[word]
          word_embedding = np.asarray(values[1:], dtype='float32')
          embedding_matrix[word_idx] = word_embedding
    return embedding_matrix

  def _build_model(self):
    I = Input(shape=(self.hparams.max_sequence_length,), dtype='float32')
    E = Embedding(
        len(self.tokenizer.word_index) + 1,
        self.hparams.embedding_dim,
        weights=[self.embedding_matrix],
        input_length=self.hparams.max_sequence_length,
        trainable=False)(I)
    X5 = Conv1D(128, 5, activation='relu', padding='same')(E)
    X5 = MaxPooling1D(250, padding='same')(X5)
    X4 = Conv1D(128, 4, activation='relu', padding='same')(E)
    X4 = MaxPooling1D(250, padding='same')(X4)
    X3 = Conv1D(128, 3, activation='relu', padding='same')(E)
    X3 = MaxPooling1D(250, padding='same')(X3)
    X = Concatenate(axis=-1)([X5, X4, X3])
    X = Flatten()(X)
    X = Dropout(self.hparams.dropout_rate)(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(self.hparams.dropout_rate)(X)
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

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_path', type=str, default='local_data/kaggle_train.csv', help='Path to the training data.')
  parser.add_argument(
      '--test_path', type=str, default='local_data/kaggle_test.csv', help='Path to the test data.')
  parser.add_argument(
      '--embeddings_path', type=str, default='local_data/glove.6B/glove.6B.100d.txt', help='Path to the embeddings.')
  parser.add_argument(
      '--model_path', type=str, default='local_data/keras_kaggle_model.h5', help='Path to model file.')

  FLAGS, unparsed = parser.parse_known_args()

  model = AttentionToxModel(model_path=FLAGS.model_path, embeddings_path=FLAGS.embeddings_path)
  train = pd.read_csv(FLAGS.train_path)
  model.train(train)

  test_data = pd.read_csv(FLAGS.test_path)
  predictions = model.predict(test_data['comment_text'])
  model.score_auc(test_data)

  model.predict(['This sentence is benign'])
