#!/bin/bash

source "tf_trainer/common/dataset_config.sh"

python -m tf_trainer.tf_hub_classifier.run \
  --train_path=$train_path \
  --validate_path=$valid_path \
  --model_dir="tf_hub_classifier_local_model_dir" \
  --model_spec="gs://conversationai-models/resources/tfhub/universal-sentence-encoder-large-3/96e8f1d3d4d90ce86b2db128249eb8143a91db73" \
  --labels=$labels \
  --label_dtypes=$label_dtypes
