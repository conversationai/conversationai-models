# Toxic Comment Classification Kaggle Challenge

This directory is a place to play around with solutions for the [Toxic Comment Classification Kaggle challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The challenge was created by the Jigsaw Conversation AI team in December 2017
and the it ends in February 2018.

These models are meant to be simple baselines created independently from the Google infrastructure.

## To Run
1. Download the training (`train.csv`) and test (`test.csv`) data from the
[Kaggle challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).
2. Install library dependencies:
```shell
pip install -r requirements.txt
```

3. Run a model on a given class (e.g. 'toxic' or 'obscene'):

```shell
  python trainer/model.py \
         --train_data=train.csv \
         --predict_data=test.csv \
         --y_class=toxic \
         --model=bag_of_words \
         --train_steps=1000
```

## Available Models
  * `bag_of_words` - bag of words model with a learned word-embedding layer
  * `cnn` - a 2 layer ConvNet

## Data
Copies of the training and test data are available in Google Storage from the wikidetox project.

* train.csv: gs://kaggle-model-experiments/train.csv
* test.csv: gs://kaggle-model-experiments/test.csv
