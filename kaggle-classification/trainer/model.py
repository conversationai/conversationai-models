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

FLAGS = None

# Data Params
TRAIN_PERCENT = .8 # Percent of data to allocate to training
DATA_SEED = 48173 # Random seed used for splitting the data into train/test
MAX_LABEL = 2
MAX_DOCUMENT_LENGTH = 500 # Max length of each comment in words

CNNParams = namedtuple(
  'CNNParams', ['WINDOW_SIZE', 'EMBEDDING_SIZE','POOLING_WINDOW', 'POOLING_STRIDE',
                'N_FILTERS', 'FILTER_SHAPE1', 'FILTER_SHAPE2'])
cnn_values = {'WINDOW_SIZE':20, 'EMBEDDING_SIZE':20, 'POOLING_WINDOW':4,
              'POOLING_STRIDE':2, 'N_FILTERS':10}
cnn_values['FILTER_SHAPE1'] = [cnn_values['WINDOW_SIZE'], cnn_values['EMBEDDING_SIZE']]
cnn_values['FILTER_SHAPE2'] = [cnn_values['WINDOW_SIZE'], cnn_values['N_FILTERS']]
CNN_PARAMS = CNNParams(**cnn_values)

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

def cnn_model(features, labels, mode):
  """
  A 2 layer ConvNet to predict from sequence of words to a class.
  Largely stolen from:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/text_classification_cnn.py
  Returns a tf.estimator.EstimatorSpec.
  """
  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  word_vectors = tf.contrib.layers.embed_sequence(
      features[WORDS_FEATURE], vocab_size=n_words, embed_dim=
    CNN_PARAMS.EMBEDDING_SIZE)

  # Inserts a dimension of 1 into a tensor's shape.
  word_vectors = tf.expand_dims(word_vectors, 3)

  with tf.variable_scope('CNN_Layer1'):
    # Apply Convolution filtering on input sequence.
    conv1 = tf.layers.conv2d(
        word_vectors,
        filters=CNN_PARAMS.N_FILTERS,
        kernel_size=CNN_PARAMS.FILTER_SHAPE1,
        padding='VALID',
        # Add a ReLU for non linearity.
        activation=tf.nn.relu)
    # Max pooling across output of Convolution+Relu.
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=CNN_PARAMS.POOLING_WINDOW,
        strides=CNN_PARAMS.POOLING_STRIDE,
        padding='SAME')
    # Transpose matrix so that n_filters from convolution becomes width.
    pool1 = tf.transpose(pool1, [0, 1, 3, 2])
  with tf.variable_scope('CNN_Layer2'):
    # Second level of convolution filtering.
    conv2 = tf.layers.conv2d(
        pool1,
        filters=CNN_PARAMS.N_FILTERS,
        kernel_size=CNN_PARAMS.FILTER_SHAPE2,
        padding='VALID')
    # Max across each filter to get useful features for classification.
    pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

  # Apply regular WX + B and classification.
  logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)
  predicted_classes = tf.argmax(logits, 1)

  return estimator_spec_for_softmax_classification(
    logits=logits, labels=labels, mode=mode)

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

def main():
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
      max_document_length=MAX_DOCUMENT_LENGTH)

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
      model_fn = cnn_model
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

  FLAGS, unparsed = parser.parse_known_args()

  main()
