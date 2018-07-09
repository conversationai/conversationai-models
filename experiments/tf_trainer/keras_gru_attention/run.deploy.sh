#!/bin/bash

# Edit these!
MODEL_NAME=keras_gru_attention
MODEL_SAVED_PATH=gs://kaggle-model-experiments/${USER}/1531159115

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

