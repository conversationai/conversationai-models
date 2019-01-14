#!/bin/bash

source "tf_trainer/common/dataset_config.sh"

python -m tf_trainer.tf_char_cnn.run \
  --train_path=$train_path \
  --validate_path=$valid_path \
  --model_dir="tf_char_cnn_local_model_dir" \
  --labels=$labels \
  --label_dtypes=$label_dtypes
