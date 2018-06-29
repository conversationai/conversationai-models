#!/bin/bash

GCS_RESOURCES="gs://kaggle-model-experiments/resources"

python -m tf_trainer.keras_gru_attention.run \
  --train_path="${GCS_RESOURCES}/toxicity_q42017_train.tfrecord" \
  --validate_path="${GCS_RESOURCES}/toxicity_q42017_validate.tfrecord" \
  --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
  --model_dir="keras_gru_attention_local_model_dir"
