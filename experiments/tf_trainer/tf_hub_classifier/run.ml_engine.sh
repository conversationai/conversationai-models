#!/bin/bash
# This script runs one training job on Cloud MLE.


GCS_RESOURCES="gs://kaggle-model-experiments/resources"
DATETIME=`date '+%Y%m%d_%H%M%S'`
MODEL_NAME="tf_hub_classifier"
JOB_DIR=gs://kaggle-model-experiments/tf_trainer_runs/${USER}/${MODEL_NAME}/${DATETIME}

gcloud ml-engine jobs submit training tf_trainer_${MODEL_NAME}_${USER}_${DATETIME} \
    --job-dir=${JOB_DIR} \
    --runtime-version=1.10 \
    --scale-tier 'BASIC_GPU' \
    --module-name="tf_trainer.${MODEL_NAME}.run" \
    --package-path=tf_trainer \
    --python-version "3.5" \
    --region=us-east1 \
    --verbosity=debug \
    -- \
    --train_path="${GCS_RESOURCES}/toxicity_q42017_train.tfrecord" \
    --validate_path="${GCS_RESOURCES}/toxicity_q42017_validate.tfrecord" \
    --model_dir="${JOB_DIR}/model_dir" \
    --n_export=7
