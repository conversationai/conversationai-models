#!/bin/bash

#
# A script to train the kaggle model locally.
# Assumes that train.csv and test.csv are downloaded into the local_data/
# directory.
#

gcloud ml-engine local train \
     --module-name=trainer.model \
     --package-path=trainer \
     --job-dir=model -- \
     --train_data=local_data/train.csv \
     --predict_data=local_data/test.csv \
     --y_class=toxic \
     --train_steps=100
