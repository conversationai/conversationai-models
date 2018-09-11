"""Tensorflow Estimator implementation of Word Label Embeddings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_trainer.common import base_model
from typing import Set

FLAGS = tf.app.flags.FLAGS

# Hyperparameters
tf.app.flags.DEFINE_float('learning_rate', 0.000003,
                          'The learning rate to use during training.')
tf.app.flags.DEFINE_integer('embedding_size', 100,
                            'The number of dimensions in the word embedding.')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning.
tf.app.flags.DEFINE_string(
    'dense_units', '128',
    'Comma delimited string for the number of hidden units in the dense layer.')


class TFWordLabelEmbeddingModel(base_model.BaseModel):

  def __init__(self, text_feature_name: str, target_label: str) -> None:
    self._text_feature_name = text_feature_name
    self._target_label = target_label

  @staticmethod
  def hparams():
    dense_units = [int(units) for units in FLAGS.dense_units.split(',')]
    hparams = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        embedding_size=FLAGS.embedding_size,
        dense_units=dense_units)
    return hparams

  def estimator(self, model_dir):
    estimator = tf.estimator.Estimator(
        model_fn=self._model_fn,
        params=self.hparams(),
        config=tf.estimator.RunConfig(model_dir=model_dir))
    return estimator

  def _model_fn(self, features, labels, mode, params, config):
    word_emb_seq = features[self._text_feature_name]

    # Constants

    labels = labels[self._target_label]

    # Class emb
    class_emb_initializer = tf.random_normal_initializer(
        mean=0.0, stddev=1.0, dtype=tf.float32)
    class_embs = tf.get_variable(
        'class_embs', [2, params.embedding_size],
        initializer=class_emb_initializer)

    word_emb_seq_norm = tf.nn.l2_normalize(word_emb_seq, axis=-1)
    class_embs_norm = tf.nn.l2_normalize(class_embs, axis=-1)
    #cosine_similarity = tf.matmul(word_emb_seq_norm,
    #                              class_embs_norm)
    cosine_distance = tf.contrib.keras.backend.dot(
        word_emb_seq_norm, tf.transpose(class_embs_norm))
    max_cosine_distance = tf.reduce_max(cosine_distance, axis=-1)
    attention = tf.nn.softmax(max_cosine_distance, axis=-1)
    attention = tf.expand_dims(attention, axis=-1)

    weighted_word_emb = tf.reduce_sum(word_emb_seq * attention, axis=1)

    logits = weighted_word_emb
    for num_units in params.dense_units:
      logits = tf.layers.dense(
          inputs=logits, units=num_units, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=logits, units=1, activation=None)

    head = tf.contrib.estimator.binary_classification_head(
        name=self._target_label)

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    return head.create_estimator_spec(
        features=features,
        labels=labels,
        mode=mode,
        logits=logits,
        optimizer=optimizer)
