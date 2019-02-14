"""Tests for episodic_tfrecord_input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_trainer.common import episodic_tfrecord_input


class EpisodicTFRecordInputTest(tf.test.TestCase):

  def test(self):
    train_dir = 'gs://kaggle-model-experiments/resources/transfer_learning_data/many_communities_pruned_episodes'
    tf.logging.info('CREATE')
    e = episodic_tfrecord_input.EpisodicTFRecordInput(train_dir, 'asdf')
    tf.logging.info('GET DATA')
    episodic_batch = e.train_input_fn()
    with tf.Session() as session:
      tf.logging.info('FIRST BATCH')
      tf.logging.info(session.run(episodic_batch))
      tf.logging.info('SECOND BATCH')
      print(session.run(episodic_batch))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
