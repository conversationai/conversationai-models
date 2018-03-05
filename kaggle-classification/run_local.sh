#!/bin/bash

#
# A script to train the kaggle model locally
#

gcloud ml-engine local train \
     --module-name=trainer.model \
     --package-path=trainer \
     --job-dir=model -- \
     --train_data=gs://kaggle-model-experiments/train.csv \
     --predict_data=gs://kaggle-model-experiments/test.csv \
     --y_class=toxic \
     --train_steps=100
