#!/bin/bash

GCS_RESOURCES="gs://kaggle-model-experiments/resources"
DATETIME=`date '+%Y%m%d_%H%M%S'`
JOB_DIR=gs://kaggle-model-experiments/tf_trainer_runs/${USER}/${DATETIME}

gcloud ml-engine jobs submit training tf_trainer_${USER}_${DATETIME} \
    --job-dir=${JOB_DIR} \
    --runtime-version=1.8 \
    --module-name=tf_trainer.run \
    --package-path=tf_trainer \
    --region=us-east1 \
    --verbosity=debug \
    --config=tf_trainer/hparam_config.yaml \
    -- \
    --train_path="${GCS_RESOURCES}/toxicity_q42017_train.tfrecord" \
    --validate_path="${GCS_RESOURCES}/toxicity_q42017_validate.tfrecord" \
    --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
    --model_dir="${JOB_DIR}/model_dir"
