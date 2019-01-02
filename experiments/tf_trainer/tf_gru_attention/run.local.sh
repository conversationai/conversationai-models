#!/bin/bash

# Note:
# We currently use 2 different embeddings:
# - glove.6B/glove.6B.300d.txt
# - google-news/GoogleNews-vectors-negative300.txt
# Glove assumes all words are lowercased, while Google-news handles different casing.
# As there is currently no tf operation that perform lowercasing, we have the following 
# requirements:
# - For google news: Run preprocess_in_tf=True (no lowercasing).
# - For glove.6B, Run preprocess_in_tf=False (will force lowercasing).


GCS_RESOURCES="gs://kaggle-model-experiments/resources"

if [ "$1" == "civil_comments" ]; then
    
    train_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/train-*.tfrecord"
    valid_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/eval-*.tfrecord"
    labels="toxicity"
    label_dtypes="float"

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


python -m tf_trainer.tf_gru_attention.run \
  --train_path=$train_path \
  --validate_path=$valid_path \
  --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
  --model_dir="tf_gru_attention_local_model_dir" \
  --labels=$labels \
  --label_dtypes=$label_dtypes \
  --preprocess_in_tf=False



