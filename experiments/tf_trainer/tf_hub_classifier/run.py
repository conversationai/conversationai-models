"""Experiments with Toxicity Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import common flags and run code. Must be imported first.
from tf_trainer.common import model_trainer

from tf_trainer.common import tfrecord_simple
from tf_trainer.common import serving_input
from tf_trainer.common import types
from tf_trainer.tf_hub_classifier import model as tf_hub_classifier

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("text_feature_name", "comment_text",
                           "Feature name of the text feature.")
tf.app.flags.DEFINE_string("key_name", "comment_key",
                           "Name of the key feature for serving examples.")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "The batch size to use during training.")
tf.app.flags.DEFINE_integer("train_steps", 10,
                            "The number of steps to train for.")
tf.app.flags.DEFINE_integer("eval_period", 500,
                            "The number of steps per eval period.")
tf.app.flags.DEFINE_integer("eval_steps", 50,
                            "The number of steps to eval for.")

LABELS = {
    "frac_neg": tf.float32,
}  # type: Dict[str, tf.DType]


def create_serving_input_fn(text_feature_name, key_name):

  def serving_input_fn_tfrecords():

    serialized_example = tf.placeholder(
        shape=[None],
        dtype=tf.string,
        name="input_example_tensor"
    )
    feature_spec = {
        text_feature_name: tf.FixedLenFeature([], dtype=tf.string),
        key_name: tf.FixedLenFeature([], dtype=tf.int64)
    }

    features = tf.parse_example(
        serialized_example, feature_spec)

    return tf.estimator.export.ServingInputReceiver(
        features,
        serialized_example)

  return serving_input_fn_tfrecords


def main(argv):
  del argv  # unused

  dataset = tfrecord_simple.TFSimpleRecordInput(
      train_path=FLAGS.train_path,
      validate_path=FLAGS.validate_path,
      text_feature=FLAGS.text_feature_name,
      labels=LABELS,
      batch_size=FLAGS.batch_size)

  model = tf_hub_classifier.TFHubClassifierModel(
      FLAGS.text_feature_name,
      set(LABELS.keys())
      )

  trainer = model_trainer.ModelTrainer(dataset, model)
  trainer.train_with_eval(FLAGS.train_steps, FLAGS.eval_period, FLAGS.eval_steps)

  serving_input_fn = create_serving_input_fn(
      text_feature_name=FLAGS.text_feature_name,
      key_name=FLAGS.key_name)
  trainer.export(serving_input_fn)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
