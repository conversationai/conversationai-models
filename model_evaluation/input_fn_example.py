# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Defines some examples of input_fn for the evaluation notebook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import pandas as pd
import pkg_resources
import os
import random
import re

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from unintended_ml_bias import model_bias_analysis
from utils_export import utils_tfrecords

#Faster to access GCS file + https://github.com/tensorflow/tensorflow/issues/15530
os.environ['GCS_READ_CACHE_MAX_SIZE_MB'] = '0'

#TODO(fprost): Clean this file.

#### #### #### #### #### ####
#### PERFORMANCE DATASET ####
#### #### #### #### #### ####


def create_input_fn_toxicity_performance(tokenizer, model_input_comment_field):
  """Generates an input_fn to evaluate model performance on toxicity dataset."""

  TOXICITY_PERFORMANCE_DATASET = 'gs://conversationai-models/resources/toxicity_data/toxicity_q42017_test.tfrecord'
  TOXICITY_DATA_LABEL = 'frac_neg'  #Name of the label in the dataset
  TOXICITY_COMMENT_NAME = 'comment_text'  #Name of the comment in the dataset

  # DECODING
  decoding_input_features = {
      TOXICITY_COMMENT_NAME: tf.FixedLenFeature([], dtype=tf.string),
      TOXICITY_DATA_LABEL: tf.FixedLenFeature([], dtype=tf.float32)
  }

  def input_fn_performance_toxicity(max_n_examples=None,
                                    random_filter_keep_rate=1.0):
    res = utils_tfrecords.decode_tf_records_to_pandas(
        decoding_input_features, TOXICITY_PERFORMANCE_DATASET, max_n_examples,
        random_filter_keep_rate)
    res[model_input_comment_field] = list(
        map(tokenizer, res[TOXICITY_COMMENT_NAME]))
    res = res.rename(columns={TOXICITY_DATA_LABEL: 'label'})
    res['label'] = list(map(lambda x: bool(round(x)), list(res['label'])))
    final = res.copy(deep=True)
    return final

  return input_fn_performance_toxicity


#### #### #### #### ####
#### CIVIL DATASET  ####
#### #### #### #### ####

CIVIL_COMMENTS_PATH = 'gs://conversationai-models/resources/civil_comments_data/train_eval_test/eval-00000-of-00003.tfrecord'
THRESHOLD_BIAS_CIVIL = 0.5

civil_comments_spec = {
    'comment_text':
        tf.FixedLenFeature([], dtype=tf.string),
    'id':
        tf.FixedLenFeature([], dtype=tf.string),
    'toxicity':
        tf.FixedLenFeature([], dtype=tf.float32),
    'severe_toxicity':
        tf.FixedLenFeature([], dtype=tf.float32),
    'obscene':
        tf.FixedLenFeature([], dtype=tf.float32),
    'sexual_explicit':
        tf.FixedLenFeature([], dtype=tf.float32),
    'identity_attack':
        tf.FixedLenFeature([], dtype=tf.float32),
    'insult':
        tf.FixedLenFeature([], dtype=tf.float32),
    'threat':
        tf.FixedLenFeature([], dtype=tf.float32),
    'toxicity_annotator_count':
        tf.FixedLenFeature([], dtype=tf.int64),
    'identity_annotator_count':
        tf.FixedLenFeature([], dtype=tf.int64),
    'male':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'female':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'transgender':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'other_gender':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'heterosexual':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'homosexual_gay_or_lesbian':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'bisexual':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'other_sexual_orientation':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'christian':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'jewish':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'muslim':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'hindu':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'buddhist':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'atheist':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'other_religion':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'black':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'white':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'asian':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'latino':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'other_race_or_ethnicity':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'physical_disability':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'intellectual_or_learning_disability':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'psychiatric_or_mental_illness':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
    'other_disability':
        tf.FixedLenFeature([], dtype=tf.float32, default_value=-1.),
}

identity_terms_civil = [
    'male', 'female', 'transgender', 'other_gender', 'heterosexual',
    'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation',
    'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist',
    'other_religion', 'black', 'white', 'asian', 'latino',
    'other_race_or_ethnicity', 'physical_disability',
    'intellectual_or_learning_disability', 'psychiatric_or_mental_illness',
    'other_disability'
]

CIVIL_COMMENT_NAME = 'comment_text'


def create_input_fn_civil_performance(tokenizer, model_input_comment_field):
  """Generates an input_fn to evaluate model performance on civil dataset."""

  def input_fn_performance_civil(max_n_examples=None,
                                 random_filter_keep_rate=1.0):
    civil_df_raw = utils_tfrecords.decode_tf_records_to_pandas(
        civil_comments_spec,
        CIVIL_COMMENTS_PATH,
        max_n_examples=max_n_examples,
        random_filter_keep_rate=random_filter_keep_rate,
    )
    civil_df_raw[CIVIL_COMMENT_NAME] = list(
        map(tokenizer, civil_df_raw[CIVIL_COMMENT_NAME]))
    civil_df_raw['toxicity'] = list(
        map(lambda x: bool(round(x)), list(civil_df_raw['toxicity'])))
    civil_df_raw = civil_df_raw.rename(columns={
        CIVIL_COMMENT_NAME: model_input_comment_field,
        'toxicity': 'label'
    })
    res = civil_df_raw.copy(deep=True)
    return res

  return input_fn_performance_civil


def create_input_fn_civil_bias(tokenizer, model_input_comment_field):
  """"Generates an input_fn to evaluate model bias on civil dataset.

  Construction of this database such as:
      We keep only examples that have identity labels (with rule: male >=0).
      We apply the 'threshold_bias_civil' for each identity field.
      We select x% of the "background", i.e. examples that are 0 for each
      identify.

  Indeed, as the background is dominant, we want to reduce the size of the test
  set.
  """

  def filter_fn_civil(example, background_filter_keep_rate=0.1):
    if example['male'] < 0.:
      return False
    contains_one_identity = False
    for _term in identity_terms_civil:
      if example[_term] >= THRESHOLD_BIAS_CIVIL:
        contains_one_identity = True
    if contains_one_identity:
      return True
    else:
      return (random.random() < background_filter_keep_rate)

  def input_fn_bias_civil(max_n_examples=None):
    civil_df_raw = utils_tfrecords.decode_tf_records_to_pandas(
        civil_comments_spec,
        CIVIL_COMMENTS_PATH,
        max_n_examples=max_n_examples,
        filter_fn=filter_fn_civil,
    )
    civil_df_raw[CIVIL_COMMENT_NAME] = list(
        map(tokenizer, civil_df_raw[CIVIL_COMMENT_NAME]))
    for _term in identity_terms_civil:
      civil_df_raw[_term] = list(
          map(lambda x: x >= THRESHOLD_BIAS_CIVIL, list(civil_df_raw[_term])))
    civil_df_raw['toxicity'] = list(
        map(lambda x: bool(round(x)), list(civil_df_raw['toxicity'])))
    civil_df_raw = civil_df_raw.rename(columns={
        CIVIL_COMMENT_NAME: model_input_comment_field,
        'toxicity': 'label'
    })
    res = civil_df_raw.copy(deep=True)
    return res

  return input_fn_bias_civil


#### #### #### #### #### ####
####  SYNTHETIC DATASET  ####
#### #### #### #### #### ####


def create_input_fn_artificial_bias(tokenizer, model_input_comment_field):
  """Generates an input_fn to evaluate model bias on synthetic dataset."""

  def input_fn_bias(max_n_examples):

    # Loading it from it the unintended_ml_bias github.
    entire_test_bias_df = pd.read_csv(
        pkg_resources.resource_stream('unintended_ml_bias',
                                      'eval_datasets/bias_madlibs_77k.csv'))
    entire_test_bias_df['raw_text'] = entire_test_bias_df['Text']
    entire_test_bias_df['label'] = entire_test_bias_df['Label']
    entire_test_bias_df['label'] = list(
        map(lambda x: x == 'BAD', entire_test_bias_df['label']))
    entire_test_bias_df = entire_test_bias_df[['raw_text', 'label']].copy()
    identity_terms_synthetic = [
        line.strip() for line in pkg_resources.resource_stream(
            'unintended_ml_bias', 'bias_madlibs_data/adjectives_people.txt')
    ]
    model_bias_analysis.add_subgroup_columns_from_text(
        entire_test_bias_df, 'raw_text', identity_terms_synthetic)

    # Add preprocessing
    entire_test_bias_df['text'] = list(
        map(tokenizer, entire_test_bias_df['raw_text']))
    if max_n_examples:
      res = entire_test_bias_df.sample(n=max_n_examples, random_state=2018)
    else:
      res = entire_test_bias_df
    res = res.copy(deep=True)
    res = res.rename(columns={'raw_text': model_input_comment_field})
    return res

  return input_fn_bias

#### #### #### #### #### ####
####  BIASBIOS DATASET   ####
#### #### #### #### #### ####

BIASBIOS_PATH = 'gs://conversationai-models/biosbias/dataflow_dir/data-preparation-20190220165938/eval-00000-of-00003.tfrecord'

comments_spec = {
    'comment_text':
        tf.FixedLenFeature([], dtype=tf.string),
    'gender':
        tf.FixedLenFeature([], dtype=tf.string),
    'title':
        tf.FixedLenFeature([], dtype=tf.int64)
}

identity_terms = [
    'gender'
]

COMMENT_NAME = 'comment_text'
LABEL_NAME = 'title'


def create_input_fn_biasbios(tokenizer, model_input_comment_field):
  """"Generates an input_fn to evaluate model bias on civil dataset.

  Construction of this database such as:
      We keep only examples that have identity labels (with rule: male >=0).
      We apply the 'threshold_bias_civil' for each identity field.
      We select x% of the "background", i.e. examples that are 0 for each
      identify.

  Indeed, as the background is dominant, we want to reduce the size of the test
  set.
  """

  def filter_fn_biasbios(example, background_filter_keep_rate=1.0):
    return (random.random() < background_filter_keep_rate)

  def input_fn_biasbios(max_n_examples=None, random_filter_keep_rate=1.0):
    df_raw = utils_tfrecords.decode_tf_records_to_pandas(
        comments_spec,
        BIASBIOS_PATH,
        max_n_examples=max_n_examples,
        filter_fn=filter_fn_biasbios,
    )
    df_raw[COMMENT_NAME] = list(
        map(tokenizer, df_raw[COMMENT_NAME]))
    #for _term in identity_terms:
    #  df_raw[_term] = list(df_raw[_term])
    #df_raw[LABEL_NAME] = list(df_raw[LABEL_NAME])
    df_raw = df_raw.rename(columns={
        COMMENT_NAME: model_input_comment_field,
        LABEL_NAME: 'label'
    })
    res = df_raw.copy(deep=True)
    return res

  return input_fn_biasbios
