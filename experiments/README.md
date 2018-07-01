# Text Classification Framework

This directory contains an ML framework for text classification. We illustrate
it with toxic (and other attributes) comment classification.

The framework is structured as a series of common files and templates to quickly
construct models on top of the [Keras](https://keras.io/) or the [TensorFlow
Estimator API](https://www.tensorflow.org/programmers_guide/estimators).

The templates also demonstrate how these models can be trained using [Google ML
Engine](https://cloud.google.com/ml-engine/) and track experiments with
[Comet.ML](https://www.comet.ml/).


## Environment Setup

### Python Dependencies

Install library dependencies (it is optional, but recommended to install these
in a [Virtual Environment](https://docs.python.org/3/tutorial/venv.html):

    ```shell
    # The recommended python3 way (virtual environment, technically optional):
    python3 -m venv .virtualenv
    source .virtualenv/bin/activate
    # Install dependencies
    pip install -r requirements.txt
    ```

### Cloud and ML Engine configuration

TODO(nthain)

### Commet.ML configuration

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
[`experiments/tf_trainer/common/convert_csv_to_tfrecord.py`](https://github.com/conversationai/conversationai-models/blob/master/experiments/tf_trainer/common/convert_csv_to_tfrecord.py)
for a simple CSV to `tf.record` converter.

If you have a [Comet ML](https://www.comet.ml/) key, you can use that platform
to monitor your model training progress and quality. Simply add your api key to
the file `comet_api_key.txt` in this directory.


## Evaluate an Existing Model on New Data

TODO(nthain)


## Development

Check the typings:

```shell
mypy --ignore-missing-imports -p tf_trainer
```

Run the tests:

```shell
python -m tf_trainer.common.tfrecord_input_test
python -m tf_trainer.common.base_keras_model_test
```

TODO(ldixon): maybe use Bazel for building/testing, so that we get
type-checking, testing, etc all done with one command.

### Building a New Model

TODO(jjtan)
