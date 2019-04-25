"""Experiments with Toxicity Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_trainer.common import base_model
from tf_trainer.common import model_trainer
from tf_trainer.common import serving_input
from tf_trainer.common import bert_episodic_input
from tf_trainer.tf_prototypical_network import model

import bert
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify outpit directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

model_fn = model_fn_builder(
    num_labels=len(label_list),
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
    model_fn=model_fn, config=run_config, params={"batch_size": BATCH_SIZE})

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)

print(f'Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)


def main(argv):
  del argv  # unused

  tokenizer = model.create_tokenizer_from_hub_module()
  train_input_examples = bert_episodic_input.BertEpisodicInput(
      "gs://kaggle-model-experiments/resources/transfer_learning_data/many_communities_pruned_episodes",
      "", tokenizer).train_input_fn(tokenizer)

  #train_features = [
  #    bert.run_classifier.convert_examples_to_features(episode_examples, [0, 1],
  #                                                     256, tokenizer)
  #    for episode_examples in train_input_examples
  #]

  with tf.Session() as sess:
    print(sess.run(train_input_examples))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
