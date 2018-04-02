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
import os.path
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from os.path import expanduser
from sklearn import metrics
from tensorflow.python.framework.errors_impl import NotFoundError
from keras_trainer.cnn_with_attention import CNNWithAttention
from keras_trainer.single_layer_cnn import SingleLayerCnn
from keras_trainer.custom_metrics import auc_roc

FLAGS = None

TEMPORARY_MODEL_PATH = 'model.h5'

VALID_MODELS = {
  'cnn_with_attention': CNNWithAttention,
  'single_layer_cnn': SingleLayerCnn
}

DEFAULT_HPARAMS = tf.contrib.training.HParams(
    learning_rate=0.00005,
    dropout_rate=0.5,
    batch_size=128,
    epochs=1,
    sequence_length=250,
    embedding_dim=100,
    train_embedding=False,
    model_type='single_layer_cnn',
    filter_sizes=[3,4,5],
    num_filters=[128,128,128],
    attention_intermediate_size=128)

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


class ModelRunner():
  """Toxicity model using CNN + Attention"""
  
  def __init__(self,
               job_dir,
               embeddings_path,
               log_path,
               hparams):
    if os.path.exists(TEMPORARY_MODEL_PATH):
      raise FileExistsError('The following file path already exists: {}'.format(TEMPORARY_MODEL_PATH))

    self.job_dir = job_dir
    self.model_path = os.path.join(job_dir, 'model.h5')
    self.embeddings_path = embeddings_path
    self.log_path = log_path
    self.hparams = hparams
    print('Setting up tokenizer...')
    self.tokenizer = self._setup_tokenizer()
    print('Setting up embedding matrix...')
    self.embeddings_matrix = self._setup_embeddings_matrix()
    print('Loading model...')
    self._load_model()
  
  def train(self, train):
    if self.hparams.model_type in VALID_MODELS:
      model = VALID_MODELS[self.hparams.model_type](self.embeddings_matrix, self.hparams).get_model()
    else:
      raise ValueError('You have specified an invalid model type.')
    train_comment = self._prep_texts(train['comment_text'])
    train_labels = np.array(list(zip(*[train[label] for label in LABELS])))

    callbacks = [
        ModelCheckpoint(
            TEMPORARY_MODEL_PATH, save_best_only=True, verbose=True),
        EarlyStopping(
            monitor='val_loss', mode='auto'),
        TensorBoard(self.log_path)
    ]

    model.fit(
        x=train_comment, y=train_labels,
        batch_size=self.hparams.batch_size,
        epochs=self.hparams.epochs,
        validation_split=0.1,
        callbacks=callbacks)

    # Necessary because we can't save h5 files to cloud storage directly via
    # Checkpoint.
    tf.gfile.MakeDirs(self.job_dir)
    tf.gfile.Copy(TEMPORARY_MODEL_PATH, self.model_path, overwrite=True)
    tf.gfile.Remove(TEMPORARY_MODEL_PATH)
    print('Saved model to {}'.format(self.model_path))
    
    self._load_model()

  def predict(self, texts):
    data = self._prep_texts(texts)
    return self.model.predict(data)

  def score_auc(self, data):
    predictions = self.predict(data['comment_text'])
    # Get an array where each element is a list of all the labels for the
    # specific instance.
    labels = np.array(list(zip(*[data[label] for label in LABELS])))
    individual_auc_scores = metrics.roc_auc_score(labels, predictions, average=None)
    print("Individual AUCs: {}".format(list(zip(LABELS, individual_auc_scores))))
    mean_auc_score = metrics.roc_auc_score(labels, predictions, average='macro')
    print("Mean AUC: {}".format(mean_auc_score))

  def _prep_texts(self, texts):
    return pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=self.hparams.sequence_length)

  def _load_model(self):
    try:
      tf.gfile.Copy(self.model_path, TEMPORARY_MODEL_PATH, overwrite=True)
      self.model = load_model(TEMPORARY_MODEL_PATH, custom_objects={'auc_roc': auc_roc})
      tf.gfile.Remove(TEMPORARY_MODEL_PATH)
      print('Model loaded from: {}'.format(self.model_path))
    except NotFoundError:
      print('Could not load model at: {}'.format(self.model_path))

  def _setup_tokenizer(self):
    words = []
    with tf.gfile.Open(self.embeddings_path, 'r') as f:
      for line in f:
        words.append(line.split()[0])
    tokenizer = Tokenizer(lower=True, oov_token='<unk>')
    tokenizer.fit_on_texts(words)
    self.hparams.vocab_size = len(tokenizer.word_index) + 1
    return tokenizer

  def _setup_embeddings_matrix(self):
    embeddings_matrix = np.zeros((self.hparams.vocab_size, self.hparams.embedding_dim))
    with tf.gfile.Open(self.embeddings_path, 'r') as f:
      for line in f:
        values = line.split()
        word = values[0]
        if word in self.tokenizer.word_index:
          word_idx = self.tokenizer.word_index[word]
          word_embedding = np.asarray(values[1:], dtype='float32')
          embeddings_matrix[word_idx] = word_embedding
    embeddings_matrix[self.hparams.vocab_size - 1] = embeddings_matrix.mean(axis=0)
    return embeddings_matrix

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_path', type=str, default='local_data/train.csv', help='Path to the training data.')
  parser.add_argument(
      '--validation_path', type=str, default='local_data/validation.csv', help='Path to the test data.')
  parser.add_argument(
      '--embeddings_path', type=str, default='local_data/glove.6B/glove.6B.100d.txt', help='Path to the embeddings.')
  parser.add_argument(
      '--job-dir', type=str, default='local_data/', help='Path to model file.')
  parser.add_argument(
      '--log_path', type=str, default='local_data/logs/', help='Path to write tensorboard logs.')
  parser.add_argument(
      '--hparams', type=str, default='', help='Comma separated list of "name=value" pairs.')

  FLAGS = parser.parse_args()

  hparams = DEFAULT_HPARAMS
  hparams.parse(FLAGS.hparams)

  model = ModelRunner(job_dir=FLAGS.job_dir, embeddings_path=FLAGS.embeddings_path, log_path=FLAGS.log_path, hparams=hparams)
  with tf.gfile.Open(FLAGS.train_path, 'rb') as f:
    train = pd.read_csv(f, encoding='utf-8')
  model.train(train)

  with tf.gfile.Open(FLAGS.validation_path, 'rb') as f:
    test_data = pd.read_csv(f, encoding='utf-8')
  model.score_auc(test_data)

  model.predict(['This sentence is benign'])
