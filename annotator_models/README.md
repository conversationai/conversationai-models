# Modeling Anotators

This is an implementation of the [Dawid-Skene model](http://crowdsourcing-class.org/readings/downloads/ml/EM.pdf). Dawid-Skene is an unsupervised model that can be used to improve the quality of a crowdsourced dataset by learning annotator error rate and predicting the true item labels.

This code was adapted from an [implementation](https://github.com/dallascard/dawid_skene) by [dallascard](https://github.com/dallascard).

## To Run Locally

1.  Setup a [virtualenv](https://virtualenvwrapper.readthedocs.io/en/latest/) for
    the project (recommended, but technically optional).

    Python 2:

    ```
    python -m virtualenv env
    ```

    Python 3:

    ```
    python3 -m venv env
    ```

    From either to enter your virtual env:

    ```shell
    source env/bin/activate
    ```

2.  Install library dependencies:

    ```shell
    pip install -r requirements.txt
    ```

3.  Create training data. The training data must be a CSV that has fields for
    the worker ID, item ID and label. You can specify the column names for these
    fields as flags to the training script.

    For example:
    ```
    comment_id,worker_id,toxic
    1519346288,43675129,0
    1519346288,41122119,0
    1519346288,38510102,0
    1519346288,43650017,0
    1519346288,28524232,0
    ...
    ```

4.  Run a model on a given class (e.g. 'toxic' or 'obscene'). There are examples
    of how to run the model locally and using ml-engine in [`bin/run_local`](bin/run_local) and
    [`bin/run`](bin/run) respectively.

    Note: to run in google cloud, you will need to be authenticated with
    Google Cloud (you can run `gcloud auth application-default login` to do
    this) and you must have access to the cloud bucket where the data is located
    (you can test this by running `gsutil ls  gs://kaggle-model-experiments/`).

5. The output is two files written to the `job-dir` directory specified in the run
    script.
   * `error_rates_{LABEL}_{N_ANNOTATIONS}.csv` - the error rates for each annotator
   * `predictions_{LABEL}_{N_ANNOTATIONS}.csv` - the predicted labels for each item