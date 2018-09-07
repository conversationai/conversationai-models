"""Tensorflow Estimator implementation of Word Label Embeddings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_trainer.common import base_model
from typing import Set

FLAGS = tf.app.flags.FLAGS

# Hyperparameters
# TODO: Add validation
tf.app.flags.DEFINE_float('learning_rate', 0.000003,
                          'The learning rate to use during training.')
tf.app.flags.DEFINE_float('dropout_rate', 0.3,
                          'The dropout rate to use during training.')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning.
tf.app.flags.DEFINE_string(
    'gru_units', '128',
    'Comma delimited string for the number of hidden units in the gru layer.')
tf.app.flags.DEFINE_integer('attention_units', 64,
                            'The number of hidden units in the gru layer.')
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
    gru_units = [int(units) for units in FLAGS.gru_units.split(',')]
    dense_units = [int(units) for units in FLAGS.dense_units.split(',')]
    hparams = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        dropout_rate=FLAGS.dropout_rate,
        gru_units=gru_units,
        attention_units=FLAGS.attention_units,
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
    batch_size = tf.shape(word_emb_seq)[0]
    seq_length = tf.shape(word_emb_seq)[1]

    labels = labels[self._target_label]

    # Class emb
    class_emb_initializer = tf.random_normal_initializer(
        mean=0.0, stddev=1.0, dtype=tf.float32)
    class_embs = tf.get_variable(
        'class_embs', [2, 100], initializer=class_emb_initializer)

    word_emb_seq_norm = tf.nn.l2_normalize(word_emb_seq, axis=-1)
    class_embs_norm = tf.nn.l2_normalize(class_embs, axis=-1)
    #cosine_similarity = tf.matmul(word_emb_seq_norm,
    #                              class_embs_norm)
    cosine_distance = tf.contrib.keras.backend.dot(
        word_emb_seq_norm, tf.transpose(class_embs_norm))
    max_cosine_distance = tf.reduce_max(cosine_distance, axis=-1)
    attention = tf.nn.softmax(max_cosine_distance, axis=-1)
    attention = tf.expand_dims(attention, axis=-1)

    #word_emb_seq = tf.Print(
    #    word_emb_seq,
    #    [
    #        tf.shape(class_embs),  # [2 100]
    #        tf.shape(labels),  # [32 1]
    #        tf.shape(word_emb_seq),  # [32 ? 100]
    #        tf.shape(cosine_distance),  #
    #        tf.shape(max_cosine_distance)  #
    #    ])

    weighted_word_emb = tf.reduce_sum(word_emb_seq * attention, axis=1)
    logits = tf.layers.dense(
        inputs=weighted_word_emb, units=64, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=logits, units=1, activation=None)
    #logits = tf.layers.dense(inputs=logits, units=1, activation=None)

    head = tf.contrib.estimator.binary_classification_head(
        name=self._target_label)

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    return head.create_estimator_spec(
        features=features,
        labels=labels,
        mode=mode,
        logits=logits,
        optimizer=optimizer)
