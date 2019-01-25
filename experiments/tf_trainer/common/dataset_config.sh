#!/bin/bash

BASE_PATH="gs://conversationai-models"
GCS_RESOURCES="${BASE_PATH}/resources"
MODEL_PARENT_DIR="${BASE_PATH}/tf_trainer_runs"

if [ "$1" == "civil_comments" ]; then
    train_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/train-*.tfrecord"
    valid_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/eval-*.tfrecord"
    labels="toxicity"
    label_dtypes="float"

elif [ "$1" == "toxicity" ]; then
    train_path="${GCS_RESOURCES}/toxicity_data/toxicity_q42017_train.tfrecord"
    valid_path="${GCS_RESOURCES}/toxicity_data/toxicity_q42017_validate.tfrecord"
    labels="frac_neg"
    label_dtypes="float"

elif [ "$1" == "many_communities" ]; then
    train_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_train.tfrecord"
    valid_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_validate.tfrecord"
    labels="removed"
    # removed is a boolean variable cast as an int.
    # 1 means that the comment was removed and 0 means it was not.
    label_dtypes="int"

else
    echo "First positional arg must be one of civil_comments, toxicity, many_communities."
    exit 1
fi
