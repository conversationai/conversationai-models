"""Experiments with Toxicity Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import common flags and run code. Must be imported first.
from tf_trainer.common import model_trainer

from tf_trainer.common import tfrecord_input
from tf_trainer.common import serving_input
from tf_trainer.common import text_preprocessor
from tf_trainer.common import types
from tf_trainer.keras_gru_attention import model as keras_gru_attention

import tensorflow as tf

from typing import Dict

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("embeddings_path",
                           "local_data/glove.6B/glove.6B.100d.txt",
                           "Path to the embeddings file.")
tf.app.flags.DEFINE_string("text_feature_name", "comment_text",
                           "Feature name of the text feature.")
tf.app.flags.DEFINE_string("key_name", "comment_key",
                           "Name of the key feature for serving examples.")
tf.app.flags.DEFINE_string("preprocess_in_tf", True,
                           "Run preprocessing with TensorFlow operations,"
                           "required for serving.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "The batch size to use during training.")
tf.app.flags.DEFINE_integer("train_steps", 50,
                            "The number of steps to train for.")
tf.app.flags.DEFINE_integer("eval_period", 40,
                            "The number of steps per eval period.")
tf.app.flags.DEFINE_integer("eval_steps", 20,
                            "The number of steps to eval for.")

# TODO: Missing fields are not handled properly yet.
LABELS = {
    "frac_neg": tf.float32,
    #"frac_very_neg": tf.float32
}  # type: Dict[str, tf.DType]


def main(argv):
  del argv  # unused

  embeddings_path = FLAGS.embeddings_path
  text_feature_name = FLAGS.text_feature_name
  key_name = FLAGS.key_name

  preprocessor = text_preprocessor.TextPreprocessor(embeddings_path)
  if FLAGS.preprocess_in_tf:
    tokenize_op_init = lambda: preprocessor.tokenize_tensor_op_tf_func()
  else:
    tokenize_op_init = lambda: preprocessor.tokenize_tensor_op_py_func()

  dataset = tfrecord_input.TFRecordInput(
      train_path=FLAGS.train_path,
      validate_path=FLAGS.validate_path,
      text_feature=text_feature_name,
      labels=LABELS,
      feature_preprocessor_init=tokenize_op_init,
      batch_size=FLAGS.batch_size)

  # TODO: Move embedding *into* Keras model.
  model = preprocessor.add_embedding_to_model(
      keras_gru_attention.KerasRNNModel(set(LABELS.keys())), text_feature_name)

  trainer = model_trainer.ModelTrainer(dataset, model)
  trainer.train_with_eval(FLAGS.train_steps, FLAGS.eval_period, FLAGS.eval_steps)

  if FLAGS.preprocess_in_tf:
    serving_input_fn = serving_input.create_serving_input_fn(
        feature_preprocessor_init=tokenize_op_init,
        sentence_name=text_feature_name,
        key_name=key_name)
    trainer.export(serving_input_fn)


if __name__ == "__main__":
  tf.app.run(main)
