# Toxic Comment Classification Kaggle Challenge

This directory is a place to play around with solutions for the
[Toxic Comment Classification Kaggle challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
The challenge was created by the Jigsaw Conversation AI team in December 2017
and the it ends in February 2018.

These models are meant to be simple baselines created independently from the
Google infrastructure.


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

3.  For training locally, download the training (`train.csv`) and test
    (`test.csv`) data from the
    [Kaggle challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).

    If you have [a Kaggle API Key](https://github.com/Kaggle/kaggle-api#api-credentials)
    setup, you can use the [Kaggle api tool](https://github.com/Kaggle/kaggle-api)
    to download these files by running:

    ```shell
    kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p ./
    mv jigsaw-toxic-comment-classification-challenge local_data
    for z in local_data/*.zip; do unzip -x $z -d local_data/; done
    ```

    Note: the `kaggle` command is installed from the `pip` and specified in
    `requirements.txt`.

4.  Run a model on a given class (e.g. 'toxic' or 'obscene'). There are examples
    of how to run the model locally and using ml-engine in `bin/run_local` and
    `bin/run` respectively.

    Note: to run in google cloud, you will need to be authenticated with
    Google Cloud (you can run `gcloud auth application-default login` to do
    this) and you must have access to the cloud bucket where the data is located
    (you can test this by running `gsutil ls  gs://kaggle-model-experiments/`).


## Available Models
  * `bag_of_words` - bag of words model with a learned word-embedding layer
  * `cnn` - a 2 layer ConvNet


## Data

Copies of the training and test data are available in Google Storage from the
wikidetox project.

* train.csv: gs://kaggle-model-experiments/train.csv
* test.csv: gs://kaggle-model-experiments/test.csv
