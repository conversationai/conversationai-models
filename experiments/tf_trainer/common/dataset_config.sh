#!/bin/bash

BASE_PATH="gs://conversationai-models"
GCS_RESOURCES="${BASE_PATH}/resources"
MODEL_PARENT_DIR="${BASE_PATH}/tf_trainer_runs"

if [ "$1" == "civil_comments" ]; then
    train_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/train-*.tfrecord"
    valid_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/eval-*.tfrecord"
    labels="toxicity"
    label_dtypes="float"
    text_feature="comment_text"

elif [ "$1" == "toxicity" ]; then
    train_path="${GCS_RESOURCES}/toxicity_data/toxicity_q42017_train.tfrecord"
    valid_path="${GCS_RESOURCES}/toxicity_data/toxicity_q42017_validate.tfrecord"
    labels="frac_neg"
    label_dtypes="float"
    text_feature="comment_text"

elif [ "$1" == "many_communities" ]; then
    train_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_train.tfrecord"
    valid_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_validate.tfrecord"
    labels="removed"
    # removed is a boolean variable cast as an int.
    # 1 means that the comment was removed and 0 means it was not.
    label_dtypes="int"
    text_feature="comment_text"

elif [ "$1" == "many_communities_40_per_8_shot" ]; then

    if [ "$2" == "optimistic" ]; then
        train_path="${GCS_RESOURCES}/transfer_learning_data/many_communities_40_per_8_shot/augmented_train.tfrecord"
    elif [ "$2" == "pessimistic" ]; then
        train_path="${GCS_RESOURCES}/transfer_learning_data/many_communities_40_per_8_shot/original_train..tfrecord"
    else
        echo "Must provide second positional argument."
        exit 1
    fi

    valid_path="${GCS_RESOURCES}/transfer_learning_data/many_communities_40_per_8_shot/validation_query..tfrecord"
    # test_path = "${GCS_RESOURCES}/transfer_learning_data/many_communities_40_per_8_shot/test_query..tfrecord"
    labels="label"
    # removed is a boolean variable cast as an int.
    # 1 means that the comment was removed and 0 means it was not.
    label_dtypes="int"
    text_feature="text"

    # used for param tuning
    train_steps=3000
    eval_steps=250
    eval_period=200

else
    echo "First positional arg must be one of civil_comments, toxicity, many_communities."
    exit 1
fi
