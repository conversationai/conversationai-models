# Dataset preparation

This directory contains some steps to prepare our data before training our ML models.

More precisely, we want to:
 * Shuffling the data and splitting it into train, eval and test datasets.
 * Creating an artificial bias (female vs male) for our embedding experiments. This is done by modifying the toxicity rate for examples labeled as 'male'.


## Environment Setup

### Build Tools/Bazel Dependencies

Install [Bazel](https://docs.bazel.build/versions/master/install-os-x.html);
this is the build tool we use to run tests, etc.

### Python Dependencies

Install library dependencies (it is optional, but recommended to install these
in a [Virtual Environment](https://docs.python.org/3/tutorial/venv.html):

    ```shell
    # The python2 way to create and use virtual environment
    # (optional, but recommended):
    virtualenv .pyenv
    source .pyenv/bin/activate
    # Install dependencies
    pip install -r requirements.txt

    jupyter notebook

    # ... do stuff ...

    # Exit your virtual environment.
    deactivate
    ```


### Execution flow


#### Splits the data locally

We recommend using a small dataset 'train_small.tfrecord'.

  ```shell
  NOW=$(date +%Y%m%d%H%M%S)
  JOB_NAME=data-preparation-$NOW

  python run_preprocessing_data_split.py \
    --job_dir 'local_data' \
    --input_data_path 'local_data/train_small.tfrecord' \
    --output_folder 'local_data/train_eval_test/'
  ```

#### Splits the data on the cloud

  ```shell
  NOW=$(date +%Y%m%d%H%M%S)
  JOB_NAME=data-preparation-$NOW

  python run_preprocessing_data_split.py \
    --job_name $JOB_NAME \
    --job_dir gs://kaggle-model-experiments/dataflow/$JOB_NAME \
    --input_data_path 'gs://kaggle-model-experiments/resources/civil_comments_data/train.tfrecord' \
    --output_folder 'gs://kaggle-model-experiments/resources/civil_comments_data/train_eval_test' \
    --cloud
  ```

#### Creates the artificial_bias locally

```shell
  NOW=$(date +%Y%m%d%H%M%S)
  JOB_NAME=data-preparation-$NOW

  python run_preprocessing_artificial_bias.py \
    --job_dir 'local_data' \
    --input_data_path 'local_data/train_eval_test/train*.tfrecord' \
    --output_folder 'local_data/artificial_bias'
  ```

#### Creates the artificial_bias on the cloud

```shell
  NOW=$(date +%Y%m%d%H%M%S)
  JOB_NAME=data-preparation-$NOW
  python run_preprocessing_artificial_bias.py \
    --job_name $JOB_NAME \
    --job_dir gs://kaggle-model-experiments/dataflow/$JOB_NAME \
    --input_data_path 'gs://kaggle-model-experiments/resources/civil_comments_data/train_eval_test/train*.tfrecord' \
    --output_folder gs://kaggle-model-experiments/resources/civil_comments_data/artificial_bias/$NOW \
    --cloud
  ```
