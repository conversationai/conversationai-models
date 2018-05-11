#!/bin/bash

DATE=`date '+%Y%m%d_%H%M%S'`
OUTPUT_PATH=runs/${DATE}
INPUT_PATH=local_data
LOG_PATH=${OUTPUT_PATH}/logs/
COMET_KEY_FILE='comet_api_key.txt'
COMET_KEY=$(cat ${COMET_KEY_FILE})

echo "You can view the tensorboard for this job with the command:"
echo ""
echo -e "\t tensorboard --logdir=${LOG_PATH}"
echo ""
echo "And on your browser navigate to:"
echo ""
echo -e "\t http://localhost:6006/#scalars"
echo ""
echo "This will populate after a model checkpoint is saved."
echo ""

python -m keras_trainer.model \
	--train_path=${INPUT_PATH}/train.csv \
	--validation_path=${INPUT_PATH}/validation.csv \
	--embeddings_path=${INPUT_PATH}/glove.6B/glove.6B.100d.txt \
	--job-dir=${OUTPUT_PATH} \
	--log_path=${LOG_PATH} \
	--comet_key ${COMET_KEY} \
  --model_type=rnn