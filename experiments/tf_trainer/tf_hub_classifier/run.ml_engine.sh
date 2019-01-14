#!/bin/bash
# This script runs one training job on Cloud MLE.

source "tf_trainer/common/dataset_config.sh"
DATETIME=$(date '+%Y%m%d_%H%M%S')
MODEL_NAME="tf_hub_classifier"
MODEL_NAME_DATA="${MODEL_NAME}_$1"
JOB_DIR="${MODEL_PARENT_DIR}/${USER}/${MODEL_NAME_DATA}/${DATETIME}"

gcloud ml-engine jobs submit training tf_trainer_${MODEL_NAME_DATA}_${USER}_${DATETIME} \
    --job-dir=${JOB_DIR} \
    --runtime-version=1.10 \
    --scale-tier 'BASIC_GPU' \
    --module-name="tf_trainer.${MODEL_NAME}.run" \
    --package-path=tf_trainer \
    --python-version "3.5" \
    --region=us-east1 \
    --verbosity=debug \
    -- \
    --train_path=$train_path \
    --validate_path=$valid_path \
    --model_dir="${JOB_DIR}/model_dir" \
    --labels=$labels \
    --label_dtypes=$label_dtypes
