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
# - For glove.6B, Run preprocess_in_tf=False (will force lowercasing).

source "tf_trainer/common/dataset_config.sh"
DATETIME=$(date '+%Y%m%d_%H%M%S')
MODEL_NAME="tf_gru_attention"
MODEL_NAME_DATA=${MODEL_NAME}_$1_glove
JOB_DIR="${MODEL_PARENT_DIR}/${USER}/${MODEL_NAME_DATA}/${DATETIME}"

if [ "$1" == "civil_comments" ]; then
    batch_size=128
    attention_units=32
    dropout_rate=0.60960359286224075
    learning_rate=0.0010256671195808884
    dense_units='128'
    gru_units='128,128'
    train_steps=50000
    eval_period=1000
    eval_steps=2000
    config="tf_trainer/common/basic_gpu_config.yaml"

elif [ "$1" == "toxicity" ]; then
    batch_size=32
    attention_units=32
    dropout_rate=0.69999994803861521
    learning_rate=0.00030340058446715442
    dense_units='128'
    gru_units='128,128'
    train_steps=250000
    eval_period=1000
    eval_steps=6000
    config="tf_trainer/common/basic_gpu_config.yaml"

elif [ "$1" == "many_communities" ]; then
    batch_size=128
    attention_units=32
    dropout_rate=0.38471142580880757
    learning_rate=0.000755324856537066
    dense_units='128'
    gru_units='128'
    train_steps=700000
    eval_period=4000
    eval_steps=45000
    config="tf_trainer/common/p100_config.yaml"

else
    echo "First positional arg must be one of civil_comments, toxicity, many_communities."
    return;
fi

gcloud ml-engine jobs submit training tf_trainer_${MODEL_NAME_DATA}_${USER}_${DATETIME} \
    --job-dir=${JOB_DIR} \
    --runtime-version=1.10 \
    --config $config \
    --module-name="tf_trainer.${MODEL_NAME}.run" \
    --package-path=tf_trainer \
    --region=us-east1 \
    --verbosity=debug \
    -- \
    --train_path=$train_path \
    --validate_path=$valid_path \
    --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
    --model_dir="${JOB_DIR}/model_dir" \
    --labels=$labels \
    --label_dtypes=$label_dtypes \
    --preprocess_in_tf=False \
    --batch_size=$batch_size \
    --attention_units=$attention_units \
    --dropout_rate=$dropout_rate \
    --learning_rate=$learning_rate \
    --dense_units=$dense_units \
    --gru_units=$gru_units \
    --train_steps=$train_steps \
    --eval_period=$eval_period \
    --eval_steps=$eval_steps
