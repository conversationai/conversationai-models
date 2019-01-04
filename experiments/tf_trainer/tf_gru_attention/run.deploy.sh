#!/bin/bash
# Deploys a saved model on Cloud MLE.

if [ "$1" == "civil_comments" ] || [ "$1" == "toxicity" ] || [ "$1" == "many_communities" ] ; then
    
    MODEL_NAME=tf_gru_attention_$1_glove

else
    echo "First positional arg must be one of civil_comments, toxicity, many_communities."
    exit 1
fi


# By default, the model is the last one from the user.
MODEL_SAVED_PATH=$(gsutil ls gs://conversationai-models/tf_trainer_runs/${USER}/${MODEL_NAME}/ | tail -1)

# Create a new model.
# Will raise an error if the model already exists.
gcloud ml-engine models create $MODEL_NAME \
  --regions us-central1

# Deploy a model version.
MODEL_VERSION=v_$(date +"%Y%m%d_%H%M%S")
gcloud ml-engine versions create $MODEL_VERSION \
  --model $MODEL_NAME \
  --origin $MODEL_SAVED_PATH \
  --runtime-version 1.10
