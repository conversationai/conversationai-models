"""
Classifiers for the Toxic Comment Classification Kaggle challenge,
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

To run locally:
  python trainer/model.py --train_data=train.csv --predict_data=test.csv --y_class=toxic

To run locally using Cloud ML Engine:
  gcloud ml-engine local train \
        --module-name=trainer.model \
        --package-path=trainer \
        --job-dir=model -- \
        --train_data=train.csv \
        --predict_data=test.csv \
        --y_class=toxic \
        --train_steps=100

To run TensorBoard locally:
  tensorboard --logdir=model/

Then visit http://localhost:6006/ to see the dashboard.
"""

from __future__ import print_function
from __future__ import division

import argparse
import os
import sys
import shutil
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from trainer import wikidata
from collections import namedtuple

from tensorflow.contrib.training.python.training import hparam


FLAGS = None

# Data Params
TRAIN_PERCENT = .8 # Percent of data to allocate to training
DATA_SEED = 48173 # Random seed used for splitting the data into train/test
MAX_LABEL = 2
MAX_DOCUMENT_LENGTH = 500 # Max length of each comment in words

# CNN parameters
CNNParams = namedtuple(
  'CNNParams',['EMBEDDING_SIZE','N_FILTERS', 'FILTER_SIZES', 'DROPOUT_KEEP_PROB'])
CNN_PARAMS = CNNParams(
  EMBEDDING_SIZE=50, N_FILTERS=10, FILTER_SIZES=[2,3,4,5],  DROPOUT_KEEP_PROB=.75)

# Bag of Word parameters
BOWParams = namedtuple('BOWParams', ['EMBEDDING_SIZE'])
BOW_PARAMS = BOWParams(EMBEDDING_SIZE = 20)

WORDS_FEATURE = 'words' # Name of the input words feature.
MODEL_LIST = ['bag_of_words', 'cnn'] # Possible models

# Training Params
TRAIN_SEED = 9812 # Random seed used to initialize training
LEARNING_RATE = 0.01
BATCH_SIZE = 20

def estimator_spec_for_softmax_classification(logits, labels, mode):
  """
  Depending on the value of mode, different EstimatorSpec arguments are required.

  For mode == ModeKeys.TRAIN: required fields are loss and train_op.
  For mode == ModeKeys.EVAL: required field is loss.
  For mode == ModeKeys.PREDICT: required fields are predictions.

  Returns EstimatorSpec instance for softmax classification.
  """
  predicted_classes = tf.argmax(logits, axis=1)
  predicted_probs = tf.nn.softmax(logits, name='softmax_tensor')

  predictions = {
    # Holds the raw logit values
    'logits': logits,

    # Holds the class id (0,1) representing the model's prediction of the most
    # likely species for this example.
    'classes': predicted_classes,

    # Holds the probabilities for each prediction
    'probs': predicted_probs,
  }

  # Represents an output of a model that can be served.
  export_outputs = {
    'output': tf.estimator.export.ClassificationOutput(scores=predicted_probs)
  }

  # PREDICT Mode
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      export_outputs=export_outputs
    )

  # Calculate loss for both TRAIN and EVAL modes
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  eval_metric_ops = {
    'accuracy': tf.metrics.accuracy(
      labels=labels, predictions=predicted_classes, name='acc_op'),
    'auc': tf.metrics.auc(
      labels=labels, predictions=predicted_classes, name='auc_op'),
  }

  # Add summary ops to the graph. These metrics will be tracked graphed
  # on each checkpoint by TensorBoard.
  tf.summary.scalar('accuracy', eval_metric_ops['accuracy'][1])
  tf.summary.scalar('auc', eval_metric_ops['auc'][1])

  # TRAIN Mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    logging_hook = tf.train.LoggingTensorHook(
      tensors={'loss': loss}, every_n_iter=50)

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      training_hooks=[logging_hook],
      predictions={'loss': loss},
      export_outputs=export_outputs,
      eval_metric_ops=eval_metric_ops
    )

  # EVAL Mode
  assert mode == tf.estimator.ModeKeys.EVAL

  return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        predictions=predictions,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs
      )

def get_cnn_model(embedding_size):
  def cnn_model(features, labels, mode):
    filter_sizes = CNN_PARAMS.FILTER_SIZES
    num_filters = CNN_PARAMS.N_FILTERS
    dropout_keep_prob = CNN_PARAMS.DROPOUT_KEEP_PROB

    with tf.name_scope("embedding"):
      W = tf.Variable(
          tf.random_uniform([n_words, embedding_size], -1.0, 1.0),
          name="W")

      embedded_chars = tf.nn.embedding_lookup(W, features[WORDS_FEATURE])
      embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
      with tf.name_scope("conv-maxpool-%s" % filter_size):

        # Convolution Layer
          filter_shape = [filter_size, embedding_size, 1, num_filters]
          W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
          b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
          conv = tf.nn.conv2d(
            embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
          # Apply nonlinearity
          hh = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

          # Max-pooling over the outputs. Max over samples in batch and
          # all filters.
          pooled = tf.nn.max_pool(
            hh,
            ksize=[1, MAX_DOCUMENT_LENGTH - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")

          pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add dropout in training
    with tf.name_scope("dropout"):
      # Set dropout rate to 1 (disable dropout) by default
      h_drop = tf.nn.dropout(h_pool_flat, 1.0)

      if mode == tf.estimator.ModeKeys.TRAIN:
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    # Add a fully connected layer to do prediction
    with tf.name_scope("output"):
      W = tf.Variable(tf.truncated_normal([num_filters_total, MAX_LABEL], stddev=0.1), name="W")
      b = tf.Variable(tf.constant(0.1, shape=[MAX_LABEL]), name="b")
      scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")

    return estimator_spec_for_softmax_classification(
      logits=scores, labels=labels, mode=mode)
  return cnn_model

def bag_of_words_model(features, labels, mode):
  """
  A bag-of-words model using a learned word embedding. Note it disregards the
  word order in the text.
  Returns a tf.estimator.EstimatorSpec.
  """

  bow_column = tf.feature_column.categorical_column_with_identity(
      WORDS_FEATURE, num_buckets=n_words)

  # The embedding values are initialized randomly, and are trained along with
  # all other model parameters to minimize the training loss.
  bow_embedding_column = tf.feature_column.embedding_column(
      bow_column, dimension=BOW_PARAMS.EMBEDDING_SIZE)

  bow = tf.feature_column.input_layer(
      features,
      feature_columns=[bow_embedding_column])

  logits = tf.layers.dense(bow, MAX_LABEL, activation=None)

  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)

def main(FLAGS):
    global n_words

    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.verbose:
      tf.logging.info('Running in verbose mode')
      tf.logging.set_verbosity(tf.logging.DEBUG)

    if os.path.isdir(FLAGS.model_dir):
      tf.logging.info("Removing model data from '/{0}'".format(FLAGS.model_dir))
      shutil.rmtree(FLAGS.model_dir)

    # Load and split data
    tf.logging.info('Loading data from {0}'.format(FLAGS.train_data))

    data = wikidata.WikiData(
      FLAGS.train_data, FLAGS.y_class, seed=DATA_SEED, train_percent=TRAIN_PERCENT,
      max_document_length=MAX_DOCUMENT_LENGTH, char_ngrams=FLAGS.char_ngrams,
      min_frequency=FLAGS.min_frequency)

    n_words = len(data.vocab_processor.vocabulary_)
    tf.logging.info('Total words: %d' % n_words)

    # Build model
    if FLAGS.model == 'bag_of_words':
      model_fn = bag_of_words_model

      # Subtract 1 because VocabularyProcessor outputs a word-id matrix where word
      # ids start from 1 and 0 means 'no word'. But categorical_column_with_identity
      # assumes 0-based count and uses -1 for missing word.
      data.x_train = data.x_train - 1
      data.x_test = data.x_test - 1
    elif FLAGS.model == 'cnn':
      model_fn = get_cnn_model(FLAGS.embedding_size)
    else:
      tf.logging.error("Unknown specified model '{}', must be one of {}"
                       .format(FLAGS.model, MODEL_LIST))
      raise ValueError

    classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      config=tf.contrib.learn.RunConfig(
        tf_random_seed=TRAIN_SEED,
      ),
      model_dir=FLAGS.model_dir)

    # Train model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={WORDS_FEATURE: data.x_train},
      y=data.y_train,
      batch_size=BATCH_SIZE,
      num_epochs=None, # Note: For training, set this to None, so the input_fn
                       # keeps returning data until the required number of train
                       # steps is reached.
      shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=FLAGS.train_steps)

    # Predict on held-out test data
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={WORDS_FEATURE: data.x_test},
      y=data.y_test,
      num_epochs=1,     # Note: For evaluation and prediction set this to 1,
                        # so the input_fn will iterate over the data once and
                        # then raise OutOfRangeError
      shuffle=False)
    predicted_test = classifier.predict(input_fn=test_input_fn)
    test_out = pd.DataFrame(
      [(p['classes'], p['probs'][1]) for p in predicted_test],
      columns=['y_predicted', 'prob']
    )

    # Score with sklearn and TensorFlow
    sklearn_score = metrics.accuracy_score(data.y_test, test_out['y_predicted'])
    tf_scores = classifier.evaluate(input_fn=test_input_fn)

    train_size = len(data.x_train)
    test_size = len(data.x_test)

    baseline = len(data.y_train[data.y_train==0]) / len(data.y_train)
    if baseline < .5:
      baseline = 1 - baseline

    tf.logging.info('')
    tf.logging.info('----------Evaluation on Held-Out Data---------')
    tf.logging.info('Train Size: {0} Test Size: {1}'.format(train_size, test_size))
    tf.logging.info('Baseline (class distribution): {0:f}'.format(baseline))
    tf.logging.info('Accuracy (sklearn): {0:f}'.format(sklearn_score))

    for key in sorted(tf_scores):
      tf.logging.info("%s: %s" % (key, tf_scores[key]))

    # Export the model
    feature_spec = {
      WORDS_FEATURE: tf.FixedLenFeature(
        dtype=tf.int64, shape=MAX_DOCUMENT_LENGTH)
    }
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    classifier.export_savedmodel(FLAGS.saved_model_dir, serving_input_fn)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--verbose', help='Run in verbose mode.', action='store_true')
  parser.add_argument(
    "--train_data", type=str, default="", help="Path to the training data.")
  parser.add_argument(
    "--model_dir", type=str, default="model", help="Temp place for model files")
  parser.add_argument(
    "--saved_model_dir", type=str, default="saved_models", help="Place to saved model files")
  parser.add_argument(
      "--y_class", type=str, default="toxic",
    help="Class to train model against, one of cnn, bag_of_words")
  parser.add_argument(
      "--model", type=str, default="bag_of_words",
    help="The model to train, one of {}".format(MODEL_LIST))
  parser.add_argument(
    "--train_steps", type=int, default=100, help="The number of steps to train the model")
  parser.add_argument(
    "--embedding_size", type=int, default=50, help="The size of the word embedding")
  parser.add_argument(
    "--job-dir", type=str, default="", help="The directory where the job is staged")
  parser.add_argument(
    "--char_ngrams", type=int, default=0,
    help="Size of overlapping character ngrams to split into, use words if 0")
  parser.add_argument(
    "--min_frequency", type=int, default=0,
    help="Minimum count for tokens passed to VocabularyProcessor")

  FLAGS = parser.parse_args()


  main(FLAGS)
