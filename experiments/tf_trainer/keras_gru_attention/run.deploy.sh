#!/bin/bash
# Deploys a saved model on Cloud MLE.

# Edit these!
MODEL_NAME=keras_gru_attention
# By default, the model is the last one from the user.
MODEL_SAVED_PATH=$(gsutil ls gs://kaggle-model-experiments/tf_trainer_runs/${USER}/${MODEL_NAME}/ | tail -1)

# Create a new model.
# Will raise an error if the model already exists.
gcloud ml-engine models create $MODEL_NAME \
  --regions us-central1

# Deploy a model version.
MODEL_VERSION=v_$(date +"%Y%m%d_%H%M%S")
gcloud ml-engine versions create $MODEL_VERSION \
  --model $MODEL_NAME \
  --origin $MODEL_SAVED_PATH \
  --runtime-version 1.8 
