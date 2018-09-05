#!/bin/bash
# This script runs one training job on Cloud MLE.

# Note:
# We currently use 2 different embeddings:
# - glove.6B/glove.6B.300d.txt
# - google-news/GoogleNews-vectors-negative300.txt
# Glove assumes all words are lowercased, while Google-news handles different casing.
# As there is currently no tf operation that perform lowercasing, we have the following 
# requirements:
# - For google news: Run preprocess_in_tf=True (no lowercasing).
# - For glove.6B, Run preprocess_in_tf=Falses (will force lowercasing).

GCS_RESOURCES="gs://kaggle-model-experiments/resources"
DATETIME=`date '+%Y%m%d_%H%M%S'`
MODEL_NAME="keras_gru_attention"
JOB_DIR=gs://kaggle-model-experiments/tf_trainer_runs/${USER}/${MODEL_NAME}/${DATETIME}
LOCAL_COMET_API_KEY_FILE="comet_api_key.txt"
REMOTE_COMET_API_KEY_FILE=${GCS_RESOURCES}/${USER}_comet_api_key.txt

gsutil cp ${LOCAL_COMET_API_KEY_FILE} ${REMOTE_COMET_API_KEY_FILE} &&
gsutil acl set private ${REMOTE_COMET_API_KEY_FILE} &&
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
    --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
    --model_dir="${JOB_DIR}/model_dir" \
    --comet_key_file="${REMOTE_COMET_API_KEY_FILE}" \
    --comet_team_name="jigsaw" \
    --comet_project_name="experiments_${USER}"
