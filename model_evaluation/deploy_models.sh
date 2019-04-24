#!/bin/bash

MODEL_DIRS='gs://conversationai-models/tf_trainer_runs/fprost/tf_gru_attention_multiclass_biosbias_glove/20190328_103329/model_dir,'\
'gs://conversationai-models/tf_trainer_runs/fprost/tf_gru_attention_multiclass_biosbias_glove/20190328_103300/model_dir,'\
'gs://conversationai-models/tf_trainer_runs/fprost/tf_gru_attention_multiclass_biosbias_glove/20190328_103254/model_dir,'\
'gs://conversationai-models/tf_trainer_runs/fprost/tf_gru_attention_multiclass_biosbias_glove/20190328_103245/model_dir,'\
'gs://conversationai-models/tf_trainer_runs/fprost/tf_gru_attention_multiclass_biosbias_glove/20190328_103232/model_dir,'\
'gs://conversationai-models/tf_trainer_runs/fprost/tf_gru_attention_multiclass_biosbias_glove/20190328_103209/model_dir,'\
'gs://conversationai-models/tf_trainer_runs/fprost/tf_gru_attention_multiclass_biosbias_glove/20190328_103152/model_dir,'\
'gs://conversationai-models/tf_trainer_runs/fprost/tf_gru_attention_multiclass_biosbias_glove/20190328_103117/model_dir'


python utils_export/deploy_list_models.py --list_model_dir=$MODEL_DIRS --model_name 'tf_test_fprost'
