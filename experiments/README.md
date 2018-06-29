# Toxic Classification Framework

This directory contains a framework and a range of models for the classification of toxic comments and other attributes of online conversation. We have structured the framework as a series of common files and templates to quickly construct models on top of the Keras and tf.estimator APIs. The templates also demonstrate how these models can be trained using [Google ML Engine](https://cloud.google.com/ml-engine/) and evaluated using comet.ml.

## Environment Setup

Install library dependencies (it is optional, but recommended to install these in a [virtual environment](https://docs.python.org/3/tutorial/venv.html):

    ```shell
    # The recommended python3 way (virtual environment, technically optional):
    python3 -m venv .virtualenv
    source .virtualenv/bin/activate
    # Install dependencies
    pip install -r requirements.txt
    ```

TOCONSIDER:
 * Rename model_runner to model_trainer (it trains models)

## Training an Existing Model

To train an existing model,  simply execute the command `./tf_trainer/MODEL_NAME/run.sh` (to run locally) or the command `./tf_trainer/MODEL_NAME/run_ml_engine.sh` to run training on [Google ML Engine](https://cloud.google.com/ml-engine/).

These scripts assume that you have access to the resources on our cloud projects. If you don't, you can still run the models locally, but will have to modify the data paths in `run.sh`. At the moment, we only support reading data in tf.record format.

If you have a [Comet ML](https://www.comet.ml/) key, you can use that platform to monitor your model training progress and quality. Simply add your api key to the file `comet_api_key.txt` in this directory.

## Evaluate an Existing Model on New Data

TODO(nthain)

## Building a New Model

TODO(jjtan)
