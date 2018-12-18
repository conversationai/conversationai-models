"""Tensorflow Estimator Character CNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import layers
from tf_trainer.common import base_model
from typing import Set

FLAGS = tf.app.flags.FLAGS

# Hyperparameters
# TODO: Add validation
tf.app.flags.DEFINE_float('learning_rate', 0.00001,
                          'The learning rate to use during training.')
tf.app.flags.DEFINE_float('dropout_rate', 0.1,
                          'The dropout rate to use during training.')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning.
# TODO: add link to relevant public issue/bug/documentation?
tf.app.flags.DEFINE_string(
    'filter_sizes', '5',
    'Comma delimited string for the sizes of convolution filters.')
tf.app.flags.DEFINE_integer(
    'num_filters', 32,
    'Number of convolutional filters for every convolutional layer.')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning.
# TODO: add link to relevant public issue/bug/documentation?
tf.app.flags.DEFINE_string(
    'dense_units', '64',
    'Comma delimited string for the number of hidden units in the dense layer.')
tf.app.flags.DEFINE_integer(
    'embedding_size', 100,
    'The number of dimensions in the character embedding.')
tf.app.flags.DEFINE_string('pooling_type', 'average',
                           'Average or max pooling.')
tf.app.flags.DEFINE_integer('string_len', 1500,
                            'The length to truncate or pad to.')


class TFCharCNNModel(base_model.BaseModel):
  """TF Character CNN Model

  TF implementation of a Character CNN. Inputs should be strings.
  """

  def __init__(self, target_labels: Set[str]) -> None:
    self._target_labels = target_labels

  @staticmethod
  def hparams():
    filter_sizes = [int(units) for units in FLAGS.filter_sizes.split(',')]
    dense_units = [int(units) for units in FLAGS.dense_units.split(',')]
    hparams = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        dropout_rate=FLAGS.dropout_rate,
        filter_sizes=filter_sizes,
        num_filters=FLAGS.num_filters,
        dense_units=dense_units,
        embedding_size=FLAGS.embedding_size,
        pooling_type=FLAGS.pooling_type,
        string_len=FLAGS.string_len)
    return hparams

  def estimator(self, model_dir):
    estimator = tf.estimator.Estimator(
        model_fn=self._model_fn,
        params=self.hparams(),
        config=tf.estimator.RunConfig(model_dir=model_dir))
    return estimator

  def _model_fn(self, features, labels, mode, params, config):
    embedding = tf.Variable(
        tf.truncated_normal([256, params.embedding_size]),
        name="char_embedding")
    texts = features[base_model.TEXT_FEATURE_KEY]
    batch_size = tf.shape(texts)[0]
    byte_ids = tf.reshape(
        tf.cast(
            tf.decode_raw(
                tf.sparse_tensor_to_dense(
                    tf.string_split(texts, ''), default_value='\0'), tf.uint8),
            tf.int32), [batch_size, -1])
    padded_ids = tf.slice(
        tf.concat(
            [byte_ids,
             tf.zeros([batch_size, params.string_len], tf.int32)],
            axis=1), [0, 0], [batch_size, params.string_len])

    inputs = tf.nn.embedding_lookup(params=embedding, ids=padded_ids)

    # Conv
    X = inputs
    for filter_size in params.filter_sizes:
      X = layers.Conv1D(
          params.num_filters, filter_size, activation='relu', padding='same')(
              X)
    if params.pooling_type == 'average':
      X = layers.GlobalAveragePooling1D()(X)
    elif params.pooling_type == 'max':
      X = layers.GlobalMaxPooling1D()(X)
    else:
      raise ValueError('Unrecognized pooling type parameter')

    # FC
    logits = X
    for num_units in params.dense_units:
      logits = tf.layers.dense(
          inputs=logits, units=num_units, activation=tf.nn.relu)
      logits = tf.layers.dropout(logits, rate=params.dropout_rate)

    logits = tf.layers.dense(
        inputs=logits, units=len(self._target_labels), activation=None)

    output_heads = [
        tf.contrib.estimator.binary_classification_head(name=name)
        for name in self._target_labels
    ]
    multihead = tf.contrib.estimator.multi_head(output_heads)

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    return multihead.create_estimator_spec(
        features=features,
        labels=labels,
        mode=mode,
        logits=logits,
        optimizer=optimizer)
