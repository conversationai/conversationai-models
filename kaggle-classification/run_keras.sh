#!/bin/bash

#
# A script to train the kaggle model remotely using ml-engine.
#
# Setup Steps:
# 1. Install the gcloud SDK
# 2. Authenticate with the GCP project you want to use, `gcloud config set project [my-project]`
# 3. Put the train and test data in Cloud Storage, `gsutil cp [DATA_FILE] gs://[BUCKET_NAME]/`
#

# Edit these!
BUCKET_NAME=kaggle-model-experiments
JOB_NAME=test_kaggle_training
REGION=us-central1

DATE=`date '+%Y%m%d_%H%M%S'`
OUTPUT_PATH=gs://${BUCKET_NAME}/model/${USER}/${DATE}
HPARAM_CONFIG=keras_hparam_config.yaml

echo "Writing to $OUTPUT_PATH"

# Remote
gcloud ml-engine jobs submit training ${JOB_NAME}_${DATE} \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.4 \
    --module-name keras-trainer.model \
    --package-path keras-trainer/ \
    --region $REGION \
    --verbosity debug \
    --config ${HPARAM_CONFIG} \
    -- \
    --train_path gs://${BUCKET_NAME}/resources/train.csv \
    --validation_path gs://${BUCKET_NAME}/resources/validation.csv \
    --embeddings_path gs://${BUCKET_NAME}/resources/glove.6B/glove.6B.100d.txt \
    --model_path $OUTPUT_PATH/final_model.h5
