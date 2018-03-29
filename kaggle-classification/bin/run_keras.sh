#!/bin/bash

#
# A script to train the kaggle model remotely using ml-engine.
#
# Setup Steps:
# 1. Install the gcloud SDK
# 2. Authenticate with the GCP project you want to use, `gcloud config set project [my-project]`
# 3. Put the train and test data in Cloud Storage,
#      `gsutil cp [DATA_FILE] gs://[BUCKET_NAME]/resources`
#

# Edit these!
BUCKET_NAME=kaggle-model-experiments
JOB_NAME=${USER}_kaggle_training
REGION=us-central1

INPUT_PATH=gs://${BUCKET_NAME}/resources
DATE=`date '+%Y%m%d_%H%M%S'`
OUTPUT_PATH=gs://${BUCKET_NAME}/keras_runs/${USER}/${DATE}
LOG_PATH=${OUTPUT_PATH}/logs/
HPARAM_CONFIG=keras_hparam_config.yaml

echo "Writing to $OUTPUT_PATH"

# Remote
gcloud ml-engine jobs submit training ${JOB_NAME}_${DATE} \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.4 \
    --module-name keras_trainer.model \
    --package-path keras_trainer \
    --region $REGION \
    --verbosity debug \
    --config ${HPARAM_CONFIG} \
    -- \
    --train_path ${INPUT_PATH}/train.csv \
    --validation_path ${INPUT_PATH}/validation.csv \
    --embeddings_path ${INPUT_PATH}/glove.6B/glove.6B.100d.txt \
    --log_path ${LOG_PATH}


echo "You can view the tensorboard for this job with the command:"
echo ""
echo -e "\t tensorboard --logdir=${LOG_PATH}"
echo ""
echo "And on your browser navigate to:"
echo ""
echo -e "\t http://localhost:6006/#scalars"
echo ""
echo "This will populate after a model checkpoint is saved."
echo ""
