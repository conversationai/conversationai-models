"""Experiments with many_communities dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_trainer.common import base_model
from tf_trainer.common import model_trainer
from tf_trainer.common import serving_input
from tf_trainer.common import tfrecord_input
from tf_trainer.tf_hub_classifier import model as tf_hub_classifier

import os
import pandas as pd
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("embeddings_path",
                           "local_data/glove.6B/glove.6B.100d.txt",
                           "Path to the embeddings file.")

tf.app.flags.DEFINE_string("tmp_results_path", None,
                           "Path to the local combined (across communities) results file.")

tf.app.flags.mark_flag_as_required("warm_start_from")
tf.app.flags.mark_flag_as_required("tmp_results_path")

def main(argv):
  del argv  # unused

  dataset = tfrecord_input.TFRecordInput()
  model = tf_hub_classifier.TFHubClassifierModel(dataset.labels())

  trainer = model_trainer.ModelTrainer(dataset, model, warm_start_from=FLAGS.warm_start_from)
  trainer.train_with_eval()

  key = ('label', 'logistic')
  predictions = list(trainer.evaluate_on_dev(predict_keys=[key]))

  valid_path_csv = FLAGS.validate_path.replace("..tfrecord", ".csv")
  df = pd.read_csv(valid_path_csv)
  labels = df['label'].values

  community = os.path.basename(FLAGS.validate_path).split("..")[0]

  assert len(labels) == len(predictions), "Labels and predictions must have the same length."

  d = {
    "label" : labels,
    "prediction": [p[key][0] for p in predictions],
    "community": [community for p in predictions],
  }

  df = pd.DataFrame(data=d)
  df.to_csv(path_or_buf=FLAGS.tmp_results_path, mode='a+', index=False, header=False)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
