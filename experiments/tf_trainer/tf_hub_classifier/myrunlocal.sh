#!/bin/bash

GCS_RESOURCES="gs://kaggle-model-experiments/resources"


if [ "$1" == "civil_comments" ]; then
    
    train_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/train-*.tfrecord"
    valid_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/eval-*.tfrecord"
    labels="frac_neg,frac_very_neg,sexual_orientation,health_age_disability,gender,religion,rne,obscene,threat,insult,identity_hate,flirtation,sexual_explicit"

elif [ "$1" == "toxicity" ]; then

    train_path="${GCS_RESOURCES}/toxicity_q42017_train.tfrecord"
    valid_path="${GCS_RESOURCES}/toxicity_q42017_validate.tfrecord"
    labels="frac_neg"
    label_dtypes="float"

elif [ "$1" == "many_communities" ]; then

    train_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_train.tfrecord"
    valid_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_validate.tfrecord"
    labels="removed"
    label_dtypes="int"

else
    echo "First positional arg must be one of civil_comments, toxicity, many_communities."
    exit 1
fi


python -m tf_trainer.tf_hub_classifier.run \
  --train_path=$train_path \
  --validate_path=$valid_path \
  --model_dir="tf_hub_classifier_local_model_dir" \
  --model_spec="gs://conversationai-models/resources/tfhub/universal-sentence-ecncoder-large-3/96e8f1d3d4d90ce86b2db128249eb8143a91db73" \
  --labels=$labels \
  --label_dtypes=$label_dtypes \
  --trainable
