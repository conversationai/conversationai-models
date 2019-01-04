#!/bin/bash

source "tf_trainer/common/dataset_config.sh"

python -m tf_trainer.tf_cnn.run \
  --train_path=$train_path \
  --validate_path=$valid_path \
  --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
  --model_dir="tf_cnn_local_model_dir" \
  --labels=$labels \
  --label_dtypes=$label_dtypes
