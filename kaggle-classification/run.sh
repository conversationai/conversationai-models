#!/bin/bash

BUCKET_NAME=kaggle-model-experiments
JOB_NAME=test_kaggle_training
REGION=us-central1

DATE=`date '+%Y%m%d_%H%M%S'`
OUTPUT_PATH=gs://kaggle-model-experiments/${DATE}

echo $OUTPUT_PATH

# Remote
gcloud ml-engine jobs submit training ${JOB_NAME}_${DATE} \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.4 \
    --module-name trainer.model \
    --package-path trainer/ \
    --region $REGION \
    --verbosity debug \
    -- \
    --train_data gs://${BUCKET_NAME}/train.csv \
    --y_class toxic \
    ---predict_data gs://${BUCKET_NAME}/test.csv \
    --train_steps 1000 \
    --saved_model_dir gs://${BUCKET_NAME}/saved_models\
    --model_dir gs://${BUCKET_NAME}/model \
    --model cnn \
