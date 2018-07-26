#!/bin/bash
# This script runs one training job locally.

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

python3 -m tf_trainer.keras_gru_attention.run \
  --train_path="${GCS_RESOURCES}/toxicity_q42017_train.tfrecord" \
  --validate_path="${GCS_RESOURCES}/toxicity_q42017_validate.tfrecord" \
  --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txtt" \
  --model_dir="keras_gru_attention_local_model_dir"
