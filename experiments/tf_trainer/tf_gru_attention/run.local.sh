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

source "tf_trainer/common/dataset_config.sh"

python -m tf_trainer.tf_gru_attention.run \
  --train_path=$train_path \
  --validate_path=$valid_path \
  --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
  --model_dir="tf_gru_attention_local_model_dir" \
  --labels=$labels \
  --label_dtypes=$label_dtypes \
  --preprocess_in_tf=False