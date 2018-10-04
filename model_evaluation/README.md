# Evaluation Pipeline for Text classification models.

This directory contains utilities to use a model deployed on cloud MLE (in 'utils_export/'), and some notebooks to illustrate a typical evaluation pipeline.


## Environment Setup

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

## Evaluating one model

The notebook `jigsaw_evaluation_pipeline.ipynb` provides a example of running on evaluation metrics for the ml-fairness project.

We use the `Dataset` and `Model` utilities from `utils_export/` to interact with the models deployed on CMLE and execute the following steps:
 * Load two datasets: 1 dataset to evaluate performance (or intended bias) similar to the training data, and 1 dataset to evaluate the unintended bias that includes identity information.
 * Run the model on each dataset and collect the predictions.
 * Compute evaluation metrics: AUC on the first dataset, pinned_auc on the second one.


## Evaluating several models

This is useful to compare different training runs (with different parameters) but also to compare the evaluation metrics during the training run (several models exported during 1 training run).

TODO(fprost): Write description once the notebook is pushed


## Cloud MLE utilities

The utility library `utils_export/` intends to simplify the use of CMLE deployed models.

### Typical usage pattern

This library will handle the following "overhead" tasks:
 * Convert your pandas `DataFrame` into tf-records, adding an `example_key` to each example.
 * Send an HTTP request to CMLE to run a batch prediction job.
 * Wait for job completion.
 * Parse prediction files and join results with the initial `DataFrame` based on `example_key`.


```
  input_fn = ... (returns pandas DataFrame).
  dataset = Dataset(input_fn, dataset_dir)

  dataset.load_data(10000)

  model = Model(...)
  dataset.add_model_prediction_to_data(model)
  OR
  dataset.add_model_prediction_to_data(model, recompute=False)

  dataset.show_data()
```

### `Model`

A `Model` instance describes the key components of a CMLE model.

Key parameters are:
 * how to access the model: project_name, model_names.
 * what the expected inputs to the models are and their respective types (see EncodingFeatureSpec). The types are important to find the right encoding function for TF-records.
 * what the model outputs are.

Example:
```
model = Model(
    feature_keys_spec={'comment_text': EncodingFeatureSpec.LIST_STRING},
    prediction_keys='prediction_key',
    model_names=['model_name1:version1', 'model_name1:version2', 'model_name2:version1']
    project_name='wikidetox')
```


### `Dataset`

A `Dataset` instance is related to a pandas `DataFrame` and will be progressively augmented with the model predictions.

The dataset attributes are:
 * `input_fn`: a function that returns a `DataFrame` (input_data).
 * `DATASET_DIR`: where to save/load all the files associated with the `Dataset`, in particular input_tf_records and cloud mle predictions.
