#!/bin/bash

source "tf_trainer/common/dataset_config.sh"

python -m tf_trainer.tf_hub_classifier.run \
  --train_path=$train_path \
  --validate_path=$valid_path \
  --model_dir="tf_hub_classifier_local_model_dir" \
  --labels=$labels \
  --label_dtypes=$label_dtypes
