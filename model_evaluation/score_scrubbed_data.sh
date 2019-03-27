#!/bin/bash

MODEL_NAMES='tf_trainer_tf_gru_attention_multiclass_biosbias_glove:v_20190315_113045,'\
'tf_trainer_tf_gru_attention_multiclass_biosbias_glove:v_20190315_112954'

CLASS_NAMES='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32'
TEST_DATA='scrubbed_biasbios'
OUTPUT_PATH='gs://conversationai-models/biosbias/scored_data/scrubbed_test.csv'

echo """
Running...

python score_test_data.py \\
 --model_names=$MODEL_NAMES \\
 --class_names=$CLASS_NAMES \\
 --test_data=$TEST_DATA \\
 --output_path=$OUTPUT_PATH
"""

python score_test_data.py \
 --model_names=$MODEL_NAMES \
 --class_names=$CLASS_NAMES \
 --test_data=$TEST_DATA \
 --output_path=$OUTPUT_PATH