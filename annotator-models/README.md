# Modeling Anotators

This is an implementation of the Dawid-Skene algorithm to model annotator error rate and get predictions of true comment labels.

## To Run Locally

1.  Setup a (virtualenv)[https://virtualenvwrapper.readthedocs.io/en/latest/] for
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

3.  Create a CSV file of data to train on. The data must be a CSV that has fields:
    * `_unit_id` corresponding to the item id
    * `_worker_id` corresponding to the rater/worker id
    * `LABEL` corresponding to the label you are trying to predict

    For example:
    ```
    _unit_id,_worker_id,obscene
    1519346288,43675129,0
    1519346288,41122119,0
    1519346288,38510102,0
    1519346288,43650017,0
    1519346288,28524232,0
    ...
    ```

4.  Run a model on a given class (e.g. 'toxic' or 'obscene'). There are examples
    of how to run the model locally and using ml-engine in `bin/run_local` and
    `bin/run` respectively.

    Note: to run in google cloud, you will need to be authenticated with
    Google Cloud (you can run `gcloud auth application-default login` to do
    this) and you must have access to the cloud bucket where the data is located
    (you can test this by running `gsutil ls  gs://kaggle-model-experiments/`).
