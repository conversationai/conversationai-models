#!/bin/bash

GCS_RESOURCES="gs://kaggle-model-experiments/resources"

python -m tf_trainer.tf_hub_classifier.run \
  --train_path="${GCS_RESOURCES}/toxicity_q42017_train.tfrecord" \
  --validate_path="${GCS_RESOURCES}/toxicity_q42017_validate.tfrecord" \
  --model_dir="tf_hub_classifier_local_model_dir" \
  --labels=frac_neg,frac_very_neg,sexual_orientation,health_age_disability,gender,religion,rne,obscene,threat,insult,identity_hate,flirtation,sexual_explicit
