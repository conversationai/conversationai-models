#!/bin/bash

BASE_PATH="gs://conversationai-models"
GCS_RESOURCES="${BASE_PATH}/resources"

warm_start_from="gs://conversationai-models/tf_trainer_runs/msushkov/tf_gru_attention_many_communities_40_per_8_shot_glove/20190723_110533/model_dir"
combined_results_dir="gs://conversationai-models/resources/transfer_learning_data/many_communities_40_per_8_shot/results/tf_gru_attention/validation"

train_dir="gs://conversationai-models/resources/transfer_learning_data/many_communities_40_per_8_shot/validation_episodes/support/*.tfrecord"

eval_steps=1
eval_period=1000

labels="label"
label_dtypes="int"
text_feature="text"
    
batch_size=24
attention_units=64
dropout_rate=0.052541994248873507
dense_units='128,128'
gru_units='128'

# original, original/2, original/5, original/10, original*2
learning_rate_lst=(0.00049418814574477758 0.00024709407 0.00009883762 0.000049418814574477758 0.00098837629)
train_steps_lst=(5 10 50)

for learning_rate in "${learning_rate_lst[@]}"; do
	echo "Learning rate:"
	echo $learning_rate

	for train_steps in "${train_steps_lst[@]}"; do
		echo "Train steps:"
		echo $train_steps

		tmp_results_fname="tf_gru_attention_finetuning_baseline_trainsteps_${train_steps}_lrate_${learning_rate}.csv"
		tmp_results_path="/tmp/$tmp_results_fname"

		rm $tmp_results_path

		COUNTER=0
		for train_path in `gsutil ls $train_dir`; do
			
			valid_path=${train_path/validation_episodes\/support/validation_episodes\/query}

			rm -rf "tf_gru_attention_local_model_dir"

			python -m tf_trainer.tf_gru_attention.finetune \
			    --model_dir="tf_gru_attention_local_model_dir" \
			    --train_path=$train_path \
			    --validate_path=$valid_path \
			    --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.300d.txt" \
			    --is_embedding_trainable=False \
			    --train_steps=$train_steps \
			    --eval_period=$eval_period \
			    --eval_steps=$eval_steps \
			    --labels=$labels \
			    --label_dtypes=$label_dtypes \
			    --preprocess_in_tf=False \
			    --batch_size=$batch_size \
			    --attention_units=$attention_units \
			    --dropout_rate=$dropout_rate \
			    --learning_rate=$learning_rate \
			    --dense_units=$dense_units \
			    --gru_units=$gru_units \
			    --text_feature=$text_feature \
			    --warm_start_from=$warm_start_from \
			    --tmp_results_path=$tmp_results_path

			COUNTER=$[$COUNTER +1]

			# if [ $COUNTER -eq 2 ]
			# then
			#     break;
			# fi
		done

		gsutil cp $tmp_results_path $combined_results_dir

	done
done