# ConversationAI Models

This repository is contains example code to train machine learning models for text classification as part of the [Conversation AI](https://conversationai.github.io/) project.

# Outline of the codebase

* `experiments/` contains the ML training framework.
* `annotator-models/` contains a Dawid-Skene implementation for modelling rater quality to produce better annotations.
* `attention-colab/` contains an introductory ipython notebook for RNNs with attention, as presented at Devoxx talk ["Tensorflow, deep learning and modern RNN architectures, without a PhD by Martin Gorner"](https://www.youtube.com/watch?v=pzOzmxCR37I)
* `kaggle-classification/` early experiments with Keras and Estimator for training on [the Jigsaw Toxicity Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). Will be superceeded by `experiments/` shortly.

## About this code

This repository contains example code to help experiment with models to improve conversations; it is not an official Google product.
