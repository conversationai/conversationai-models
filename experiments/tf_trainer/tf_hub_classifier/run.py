"""Experiments with Toxicity Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_trainer.common import base_model
from tf_trainer.common import model_trainer
from tf_trainer.common import tfrecord_input
from tf_trainer.common import types
from tf_trainer.tf_hub_classifier import model as tf_hub_classifier

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("batch_size", 256,
                            "The batch size to use during training.")
tf.app.flags.DEFINE_integer("train_steps", 40000,
                            "The number of steps to train for.")
tf.app.flags.DEFINE_integer("eval_period", 1000,
                            "The number of steps per eval period.")
tf.app.flags.DEFINE_integer("eval_steps", 350,
                            "The number of steps to eval for.")


def create_serving_input_fn():

  def serving_input_fn_tfrecords():

    serialized_example = tf.placeholder(
        shape=[None],
        dtype=tf.string,
        name="input_example_tensor"
    )
    feature_spec = {
        base_model.TEXT_FEATURE_KEY: tf.FixedLenFeature([], dtype=tf.string),
        # key_name is defined in model_trainer.
        FLAGS.key_name: tf.FixedLenFeature([], dtype=tf.int64,
                                           default_value=-1)
    }

    features = tf.parse_example(
        serialized_example, feature_spec)

    return tf.estimator.export.ServingInputReceiver(
        features,
        serialized_example)

  return serving_input_fn_tfrecords


def main(argv):
  del argv  # unused

  dataset = tfrecord_input.TFRecordInput(
      batch_size=FLAGS.batch_size)

  model = tf_hub_classifier.TFHubClassifierModel(dataset.labels())

  trainer = model_trainer.ModelTrainer(dataset, model)
  trainer.train_with_eval(FLAGS.train_steps, FLAGS.eval_period, FLAGS.eval_steps)

  serving_input_fn = create_serving_input_fn()
  trainer.export(serving_input_fn)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
