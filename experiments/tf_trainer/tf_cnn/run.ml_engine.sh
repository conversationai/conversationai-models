#!/bin/bash

source "tf_trainer/common/dataset_config.sh"
DATETIME=$(date '+%Y%m%d_%H%M%S')
MODEL_NAME="tf_cnn"
MODEL_NAME_DATA=${MODEL_NAME}_$1_glove
JOB_DIR="${MODEL_PARENT_DIR}/${USER}/${MODEL_NAME_DATA}/${DATETIME}"

if [ "$1" == "civil_comments" ]; then
    batch_size=128
    dense_units='128,128'
    filter_sizes='3,4,5'
    num_filters=128
    dropout_rate=0.01527361736403272
    learning_rate=0.0001932910006772403
    pooling_type='average'
    train_steps=50000
    eval_period=1000
    eval_steps=2000

elif [ "$1" == "toxicity" ]; then
    batch_size=128
    dense_units='64'
    filter_sizes='3,4,5'
    num_filters=128
    dropout_rate=0.59761635967002524
    learning_rate=0.00028233147441192243
    pooling_type='max'
    train_steps=55000
    eval_period=1000
    eval_steps=1500

elif [ "$1" == "many_communities" ]; then
    batch_size=128
    dense_units='128,128'
    filter_sizes='3,4,5'
    num_filters=128
    dropout_rate=0.42090135248508892
    learning_rate=8.8262915612024245e-05
    pooling_type='average'
    train_steps=700000
    eval_period=4000
    eval_steps=45000

else
    echo "First positional arg must be one of civil_comments, toxicity, many_communities."
    return;
fi

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
    --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.300d.txt" \
    --model_dir="${JOB_DIR}/model_dir" \
    --is_embedding_trainable=False \
    --train_steps=$train_steps \
    --eval_period=$eval_period \
    --eval_steps=$eval_steps \
    --labels=$labels \
    --label_dtypes=$label_dtypes \
    --preprocess_in_tf=False \
    --batch_size=$batch_size \
    --dense_units=$dense_units \
    --filter_sizes=$filter_sizes \
    --num_filters=$num_filters \
    --dropout_rate=$dropout_rate \
    --learning_rate=$learning_rate \
    --pooling_type=$pooling_type

echo "Model dir:"
echo ${JOB_DIR}/model_dir
