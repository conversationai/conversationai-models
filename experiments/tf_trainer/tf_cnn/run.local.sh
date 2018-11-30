#!/bin/bash

GCS_RESOURCES="gs://kaggle-model-experiments/resources"

python -m tf_trainer.tf_cnn.run \
  --train_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/train-*.tfrecord" \
  --validate_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/eval-*.tfrecord" \
  --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
  --model_dir="tf_cnn_local_model_dir" \
  --labels="toxicity"

# python -m tf_trainer.tf_cnn.run \
#   --train_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_train.tfrecord" \
#   --validate_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_validate.tfrecord" \
#   --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
#   --model_dir="tf_cnn_local_model_dir" \
#   --labels="removed"