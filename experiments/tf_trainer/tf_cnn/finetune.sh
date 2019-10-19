#!/bin/bash

BASE_PATH="gs://conversationai-models"
GCS_RESOURCES="${BASE_PATH}/resources"

warm_start_from="gs://conversationai-models/tf_trainer_runs/msushkov/tf_cnn_many_communities_40_per_8_shot_glove/20190723_110543/model_dir"
combined_results_dir="gs://conversationai-models/resources/transfer_learning_data/many_communities_40_per_8_shot/results/tf_cnn/validation"

train_dir="gs://conversationai-models/resources/transfer_learning_data/many_communities_40_per_8_shot/validation_episodes/support/*.tfrecord"

eval_steps=1
eval_period=5

labels="label"
label_dtypes="int"
text_feature="text"
    
batch_size=24
dense_units="64,64"
filter_sizes="3,4,5"
num_filters=128
dropout_rate=0.33976339995062715
pooling_type="max"

# original, original/2, original/5, original/10, original*2
learning_rate_lst=(0.00035725183171118115 0.00017862591 0.00007145036 0.000035725183171118115 0.00071450366)
train_steps_lst=(5 10 50)

for learning_rate in "${learning_rate_lst[@]}"; do
	echo "Learning rate: $learning_rate"

	for train_steps in "${train_steps_lst[@]}"; do
		echo "Train steps: $train_steps"

		tmp_results_fname="tf_cnn_finetuning_baseline_trainsteps_${train_steps}_lrate_${learning_rate}.csv"
		tmp_results_path="/tmp/$tmp_results_fname"

		rm $tmp_results_path

		COUNTER=0
		for train_path in `gsutil ls $train_dir`; do
			echo "Community $COUNTER out of 170..."
			
			valid_path=${train_path/validation_episodes\/support/validation_episodes\/query}

			rm -rf "tf_cnn_local_model_dir"

			python -m tf_trainer.tf_cnn.finetune \
			    --model_dir="tf_cnn_local_model_dir" \
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
			    --dense_units=$dense_units \
			    --filter_sizes=$filter_sizes \
			    --num_filters=$num_filters \
			    --dropout_rate=$dropout_rate \
			    --learning_rate=$learning_rate \
			    --pooling_type=$pooling_type \
			    --text_feature=$text_feature \
			    --warm_start_from=$warm_start_from \
			    --tmp_results_path=$tmp_results_path


			COUNTER=$[$COUNTER +1]

			# if [ $COUNTER -eq 1 ]
			# then
			#     break;
			# fi
		done

		gsutil cp $tmp_results_path $combined_results_dir

	done
done