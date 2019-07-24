#!/bin/bash
# This script runs one training job on Cloud MLE.

source "tf_trainer/common/dataset_config.sh"
DATETIME=$(date '+%Y%m%d_%H%M%S')
MODEL_NAME="tf_hub_classifier"
MODEL_NAME_DATA="${MODEL_NAME}_$1"
JOB_DIR="${MODEL_PARENT_DIR}/${USER}/${MODEL_NAME_DATA}/${DATETIME}"


if [ "$1" == "civil_comments" ]; then
    batch_size=128
    dropout_rate=0.12298246947263007
    learning_rate=0.0001473127671008433
    dense_units='512,128,64'
    train_steps=50000
    eval_period=1000
    eval_steps=2000
    config="tf_trainer/common/p100_config.yaml"

elif [ "$1" == "toxicity" ]; then
    batch_size=32
    dropout_rate=0.38925458520872092
    learning_rate=0.00012916208894260696
    dense_units='512,128,64'    
    train_steps=250000
    eval_period=1000
    eval_steps=6000
    config="tf_trainer/common/p100_config.yaml"

elif [ "$1" == "many_communities" ]; then
    batch_size=128
    dropout_rate=0.6987085501984901
    learning_rate=0.00031738926545884962
    dense_units='512,128,64'    
    train_steps=700000
    eval_period=4000
    eval_steps=45000
    config="tf_trainer/common/basic_gpu_config.yaml"

elif [ "$1" == "many_communities_40_per_8_shot" ]; then

    train_steps=8000
    eval_steps=250
    eval_period=200
    config="tf_trainer/common/basic_gpu_config.yaml"

    if [ "$2" == "optimistic" ]; then

        batch_size=32
        dropout_rate=0.69999979814967772
        learning_rate=7.2549254796945835e-06
        dense_units='512,64'

    elif [ "$2" == "pessimistic" ]; then
        
        batch_size=32
        dropout_rate=0.53291173797826941
        learning_rate=6.1924912697697353e-06
        dense_units='256,128,64'

    else
        echo "Must provide second positional argument."
        exit 1
    fi

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
    --model_dir="${JOB_DIR}/model_dir" \
    --labels=$labels \
    --label_dtypes=$label_dtypes \
    --batch_size=$batch_size \
    --dropout_rate=$dropout_rate \
    --learning_rate=$learning_rate \
    --dense_units=$dense_units \
    --train_steps=$train_steps \
    --eval_period=$eval_period \
    --eval_steps=$eval_steps \
    --model_spec="gs://conversationai-models/resources/tfhub/universal-sentence-encoder-large-3/96e8f1d3d4d90ce86b2db128249eb8143a91db73" \
    --text_feature=$text_feature

