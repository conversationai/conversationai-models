#!/bin/bash

# Note:
# We currently use 2 different embeddings:
# - glove.6B/glove.6B.300d.txt
# - google-news/GoogleNews-vectors-negative300.txt
# Glove assumes all words are lowercased, while Google-news handles different casing.
# As there is currently no tf operation that perform lowercasing, we have the following 
# requirements:
# - For google news: Run preprocess_in_tf=True (no lowercasing).
# - For glove.6B, Run preprocess_in_tf=False (will force lowercasing).


GCS_RESOURCES="gs://kaggle-model-experiments/resources"

python -m tf_trainer.tf_gru_attention.run \
  --train_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/train-*.tfrecord" \
  --validate_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/eval-*.tfrecord" \
  --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
  --model_dir="tf_gru_attention_local_model_dir"