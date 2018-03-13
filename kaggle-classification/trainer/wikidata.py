"""
Class to encapsulate training and test data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn
from sklearn.model_selection import train_test_split

Y_CLASSES = ['toxic', 'severe_toxic','obscene','threat','insult','identity_hate']
TEXT_FIELD = 'comment_text'
VOCAB_PROCESSOR_FILENAME = 'vocab_processor'


class Ngrams:
  def __init__(self, ngram_size):
    self.ngram_size = ngram_size

  def __call__(self, iterator):
    """Converts a string into a list of ngrams of characters.

    list(ngrams(['abra cadabra'], 5)) =
      [('a', 'b', 'r', 'a', ' '), ('b', 'r', 'a', ' ', 'c'), ...
       ('a', 'd', 'a', 'b', 'r'), ('d', 'a', 'b', 'r', 'a')]
    """
    return (zip(*[list(x)[i:] for i in range(self.ngram_size)])
            for x in iterator)


class WikiData:

  def __init__(self, data_path, max_document_length, y_class=None,
               model_dir=None, predict_mode=False, seed=None, train_percent=None,
               char_ngrams=None, min_frequency=None):
    """
    Args:
      * data_path (string): path to file containing train or test data
      * max_document_length (int): max number of tokens per document
      * y_class (string): the class we're training or testing on
      * model_dir (string): directory with model files
      * predict_mode (boolean): true if loading data just to predict on
      * seed (integer): a random seed to use for data splitting
      * train_percent (float): the percent of data we should use for training data
    """
    data = self._load_csv(data_path)

    self.x_train, self.x_train_text = None, None
    self.x_test, self.x_test_text = None, None
    self.y_train = None
    self.y_test = None
    self.vocab_processor = None

    # -- predict mode
    if predict_mode:
      # Assume no labels, put all the data in x_test_text and x_test
      self.x_test_text = data[TEXT_FIELD]

      # Load cached VocabularyProcessor
      self.vocab_processor = self._load_vocab_processor(model_dir)

      # Process the test data
      self.x_test = np.array(list(self.vocab_processor.transform(self.x_test_text)))

      return

    # -- train mode

    # Split the data into test / train sets
    self.x_train_text, self.x_test_text, self.y_train, self.y_test \
      = self._split(data, train_percent, TEXT_FIELD, y_class, seed)

    tokenizer_fn = None
    if char_ngrams:
      tokenizer_fn=Ngrams(char_ngrams)

    # Create a VocabularyProcessor from the training data
    self.vocab_processor = tflearn.data_utils.VocabularyProcessor(
      max_document_length=max_document_length, min_frequency=min_frequency,
      tokenizer_fn=tokenizer_fn)

    self.x_train = np.array(list(self.vocab_processor.fit_transform(
      self.x_train_text)))

    # Apply the VocabularyProcessor to the test data
    self.x_test = np.array(list(self.vocab_processor.transform(
      self.x_test_text)))

  def _load_vocab_processor(self, model_dir):
    """Load a VocabularyProcessor from the provided directory"""
    vocab_processor_path = self._vocab_processor_path(model_dir)
    tf.logging.info('Loading VocabularyProcessor from {}'.format(vocab_processor_path))

    return tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_processor_path)

  def save_vocab_processor(self, model_dir):
    """Save a VocabularyProcessor in the provided directory"""
    vocab_processor_path = self._vocab_processor_path(model_dir)
    tf.logging.info('Saving VocabularyProcessor to {}'.format(vocab_processor_path))

    tf.contrib.learn.preprocessing.VocabularyProcessor.save(
      self.vocab_processor, vocab_processor_path)

  def _vocab_processor_path(self, model_dir):
    """Retruns the path to the file containing a trained VocabularyProcessor"""
    return model_dir + '/' + VOCAB_PROCESSOR_FILENAME

  def _load_csv(self, path):
    """
    Reads CSV from specified location and returns the data as a Pandas Dataframe.
    Will work with a Cloud Storage path, e.g. 'gs://<bucket>/<blob>' or a local
    path. Assumes data can fit into memory.
    """
    with tf.gfile.Open(path, 'rb') as fileobj:
      df =  pd.read_csv(fileobj, encoding='utf-8')

    return df

  def _split(self, data, train_percent, x_field, y_class, seed=None):
    """
    Split divides the Wikipedia data into test and train subsets.

    Args:
      * data (dataframe): a dataframe with data for 'comment_text' and y_class
      * train_percent (float): the fraction of data to use for training
      * x_field (string): attribute of the wiki data to use to train, e.g.
                          'comment_text'
      * y_class (string): attribute of the wiki data to predict, e.g. 'toxic'
      * seed (integer): a seed to use to split the data in a reproducible way

    Returns:
      x_train (dataframe): a pandas series with the text from each train example
      y_train (dataframe): the 0 or 1 labels for the training data
      x_test (dataframe):  a pandas series with the text from each test example
      y_test (dataframe):  the 0 or 1 labels for the test data
    """

    if y_class not in Y_CLASSES:
      tf.logging.error('Specified y_class {0} not in list of possible classes {1}'\
            .format(y_class, Y_CLASSES))
      raise ValueError

    if train_percent > 1 or train_percent < 0:
      tf.logging.error('Specified train_percent {0} is not between 0 and 1'\
            .format(train_percent))
      raise ValueError

    X = data[x_field]
    y = data[y_class]
    x_train, x_test, y_train, y_test = train_test_split(
      X, y, test_size=1-train_percent, random_state=seed)

    return x_train, x_test, np.array(y_train), np.array(y_test)
