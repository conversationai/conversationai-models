"""
Class to encapsulate training and test data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

Y_CLASSES = ['toxic', 'severe_toxic','obscene','threat','insult','identity_hate']

class WikiData:

  def __init__(self, data_path, y_class, max_document_length,
               vocab_processor_path=None, test_mode=False, seed=None,
               train_percent=None):
    """
    Args:
      * data_path (string): path to file containing train or test data
      * y_class (string): the class we're training or testing on
      * vocab_processor_path (string): if provided, the comment_text data will
          be processed with the vocab processor at that location. If not, a new
          vocab_processor will be created using the training data.
      * test_mode (boolean): true if loading data just to test on, not training
                             a model
      * seed (integer): a random seed to use for data splitting
      * train_percent (fload): the percent of data we should use for training data
    """
    data = self._load_csv(data_path)

    self.x_train, self.x_train_text = None, None
    self.x_test, self.x_test_text = None, None
    self.y_train = None
    self.y_test = None
    self.vocab_processor = None

    # If test_mode is True, then put all the data in x_test and y_test
    if test_mode:
      train_percent = 0

    # Split the data into test / train sets
    self.x_train_text, self.x_test_text, self.y_train, self.y_test \
      = self._split(data, train_percent, 'comment_text', y_class, seed)

    # Either load a VocabularyProcessor or compute one from the training data
    if test_mode:

      # If test_mode is True and no vocab_processor_path is specified, then
      # return an error. We shouldn't train a VocabProcessor at test time.
      if vocab_processor_path is None:
        tf.logging.error("Loading data in test_mode with no vocab_processor_path")
        raise ValueError

      self.vocab_processor = self.load_vocab_processor(vocab_processor_path)

    else:
      self.vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_document_length)
      self.x_train = np.array(list(self.vocab_processor.fit_transform(
        self.x_train_text)))

    # Apply the VocabularyProcessor to the test data
    self.x_test = np.array(list(self.vocab_processor.transform(
      self.x_test_text)))

  def _load_vocab_processor(self, path):
    """Load a VocabularyProcessor from the provided path"""
    return tf.contrib.learn.preprocessing.VocabularyProcessor.restore(path)

  def _load_csv(self, path):
    """
    Reads CSV from specified location and returns the data as a Pandas Dataframe.
    Will work with a Cloud Storage path, e.g. 'gs://<bucket>/<blob>' or a local
    path. Assumes data can fit into memory.
    """
    with tf.gfile.Open(path, 'r') as fileobj:
      df =  pd.read_csv(fileobj)

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
