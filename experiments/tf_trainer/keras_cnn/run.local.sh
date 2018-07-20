#!/bin/bash

GCS_RESOURCES="gs://kaggle-model-experiments/resources"
MODEL_DIR="keras_cnn_local_model_dir"

echo "You can view the tensorboard for this job with the command:"
echo ""
echo -e "\t tensorboard --logdir=${MODEL_DIR}"
echo ""
echo "And on your browser navigate to:"
echo ""
echo -e "\t http://localhost:6006/#scalars"
echo ""
echo "This will populate after a model checkpoint is saved."
echo ""


python -m tf_trainer.keras_cnn.run \
  --train_path="${GCS_RESOURCES}/toxicity_q42017_train.tfrecord" \
  --validate_path="${GCS_RESOURCES}/toxicity_q42017_validate.tfrecord" \
  --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
  --model_dir=${MODEL_DIR}

