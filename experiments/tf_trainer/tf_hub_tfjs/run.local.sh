#!/bin/bash

source "tf_trainer/common/dataset_config.sh"

python -m tf_trainer.tf_hub_tfjs.run \
  --train_path=$train_path \
  --validate_path=$valid_path \
  --model_dir="tf_hub_tfjs_local_model_dir" \
  --train_steps=9000 \
  --labels=toxicity,severe_toxicity,obscene,sexual_explicit,identity_attack,insult,threat
