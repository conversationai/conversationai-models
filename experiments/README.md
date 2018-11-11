# Text Classification Framework

This directory contains an ML framework for text classification. We illustrate
it with toxic (and other attributes) comment classification.

The framework is structured as a series of common files and templates to quickly
construct models on top of the [Keras](https://keras.io/) or the [TensorFlow
Estimator API](https://www.tensorflow.org/programmers_guide/estimators).

The templates also demonstrate how these models can be trained using [Google ML
Engine](https://cloud.google.com/ml-engine/).


## Environment Setup

### Build Tools/Bazel Dependencies

Install [Bazel](https://docs.bazel.build/versions/master/install-os-x.html);
this is the build tool we use to run tests, etc.

### Python Dependencies

Install library dependencies (it is optional, but recommended to install these
in a [Virtual Environment](https://docs.python.org/3/tutorial/venv.html):

    ```shell
    # The python3 way to create and use virtual environment
    # (optional, but recommended):
    python3 -m venv .pyenv
    source .pyenv/bin/activate
    # Install dependencies
    pip install -r requirements.txt

    # ... do stuff ...

    # Exit your virtual environment.
    deactivate
    ```

### Cloud and ML Engine configuration

TODO(nthain)


## Training an Existing Model

To train an existing model, execute either command:
 * `./tf_trainer/MODEL_NAME/run.local.sh` to run training locally, or
 * `./tf_trainer/MODEL_NAME/run.ml_engine.sh` to run training on [Google ML
Engine](https://cloud.google.com/ml-engine/).

These scripts assume that you have access to the resources on our cloud
projects. If you don't, you can still run the models locally, but will have to
modify the data paths in `run.local.sh`. At the moment, we only support reading
data in `tf.record` format. See
[`tools/convert_csv_to_tfrecord.py`](https://github.com/conversationai/conversationai-models/blob/master/experiments/tools/convert_csv_to_tfrecord.py)
for a simple CSV to `tf.record` converter.


## Running a hyper parameter tuning job

To run a hyper parameter tuning job on CMLE, execute the following command:
 * `./tf_trainer/MODEL_NAME/run.hyperparameter.sh`.

The hyperparameter configuration (MODEL_NAME/hparam_config.yaml) describes the job configuration, the parameters to tune and their respective range.

You can monitor your progress in the CMLE UI.


## Deploying a trained model on CMLE

At the end of your training, the model will be saved as a .pb file. Note: this is currently broken for keras models. TODO(fprost): Update this.

You can then deploy this model on CMLE by executing the following command:
 * `./tf_trainer/MODEL_NAME/run.deploy.sh`.

The model will be accessible as an API and available for [batch/online predictions](https://cloud.google.com/ml-engine/docs/tensorflow/batch-predict).
Further information can be found [here](https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models) about deploying models on CMLE.

## Deploying several models on CMLE for a given training run

The argument `n_export` allows you to save several models during your training run (1 model every train_steps/n).
All of the .pb filed will be saved in a subfolder of your MODEL_DIR.

There is a convenient utility in model_evaluation to help you to deploy all models on CMLE:
 * `python utils_export/deploy_continous_model.py --parent_dir MODEL_DIR --model_name MODEL_NAME `


## Evaluate an Existing Model on New Data

See `model_evaluation/` for further information.


### Type Checking

Check the typings:

```shell
mypy --ignore-missing-imports -p tf_trainer
```

It's recommended you use mypy as an additional linter in your editor.

### Testing

Run all the tests and see the output streamed:

```shell
bazel test --test_output=streamed ...
```

You can also run tests individually, directly with python like so:

```shell
python -m tf_trainer.common.tfrecord_input_test
python -m tf_trainer.common.base_keras_model_test
```

### Building a New Model

TODO(jjtan)
