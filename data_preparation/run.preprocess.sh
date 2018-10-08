#!/bin/bash

# Runs the data preprocessing in dataflow.

NOW=$(date +%Y%m%d%H%M%S)
JOB_NAME=data-preparation-$NOW

python main.py \
  --job_name $JOB_NAME \
  --job_dir gs://kaggle-model-experiments/resources/civil_comments_data/dataflow_data_$NOW \
  --input_data_path 'gs://kaggle-model-experiments/resources/civil_comments_data/train.tfrecord' \
  --cloud
