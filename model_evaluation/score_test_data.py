import getpass
import nltk
import os
import pandas as pd
import random
import tensorflow as tf

import input_fn_example
from utils_export.dataset import Dataset, Model
from utils_export import utils_cloudml
from utils_export import utils_tfrecords

tf.app.flags.DEFINE_string(
    'model_names', None, 'Comma separated list of model names deployed on ML Engine.')
tf.app.flags.DEFINE_string(
    'class_names', None, 'Comma separated list of class names to evaluate.')
tf.app.flags.DEFINE_string('test_data', None,
                           'Test data to evaluate on. Must correspond to one in input_fn_example.py.')
tf.app.flags.DEFINE_string('output_path', None,
                           'Path to write scored test data.')
tf.app.flags.DEFINE_string('project_name', 'conversationai-models',
                           'Name of GCS project.')
tf.app.flags.DEFINE_string('text_feature_name', 'tokens',
                           'Name of the text feature (see serving function call in run.py).')
tf.app.flags.DEFINE_string('sentence_key', 'comment_key',
                           'Name of input key (see serving function call in run.py).')
tf.app.flags.DEFINE_string('prediction_name', 'probabilities',
                           'Name of output prediction.')
tf.app.flags.DEFINE_integer('dataset_size', 100000,
                            'Maximum size of dataset to score.')

FLAGS = tf.app.flags.FLAGS


def get_input_fn(test_data, tokenizer, model_input_comment_field):
  if test_data == 'biasbios':
    return input_fn_example.create_input_fn_biasbios(tokenizer,
                                                     model_input_comment_field)
  elif test_data == 'scrubbed_biasbios':
    return input_fn_example.create_input_fn_biasbios(tokenizer,
                                                     model_input_comment_field,
                                                     scrubbed=True)
  else:
    raise ValueError('Dataset not currently supported.')

def tokenizer(text, lowercase=True):
  """Converts text to a list of words.

  Args:
    text: piece of text to tokenize (string).
    lowercase: whether to include lowercasing in preprocessing (boolean).
    tokenizer: Python function to tokenize the text on.

  Returns:
    A list of strings (words).
  """
  words = nltk.word_tokenize(text.decode('utf-8'))
  if lowercase:
    words = [w.lower() for w in words]
  return words



def score_data(model_names,
               class_names,
               test_data,
               output_path,
               project_name,
               text_feature_name,
               sentence_key,
               prediction_name,
               dataset_size):
  """Scores a test dataset with ML engine models and writes output as csv.

  Args:
    model_names: list of model names deployed on ML Engine.
    class_names: list of class names to evaluate.
    test_data: test data to evaluate on, must be defined in get_input_fn.
    output_path: path to write scored test data.
    project_name: name of Google Cloud project.
    text_feature_name: name of the text feature (see serving function call in run.py).
    sentence_key: name of input key (see serving function call in run.py).
    prediction_name: name of output prediction.
    dataset_size: maximum size of dataset to score.
  """
  os.environ['GCS_READ_CACHE_MAX_SIZE_MB'] = '0' #Faster to access GCS file + https://github.com/tensorflow/tensorflow/issues/15530
  nltk.download('punkt')

  model_input_spec = {
      text_feature_name: utils_tfrecords.EncodingFeatureSpec.LIST_STRING} #library will use this automatically
  model = Model(
      feature_keys_spec=model_input_spec,
      prediction_keys=prediction_name,
      example_key=sentence_key,
      model_names=model_names,
      project_name=project_name)

  input_fn = get_input_fn(test_data,
    tokenizer,
    model_input_comment_field=text_feature_name,
    )

  # Pattern for path of tf_records
  performance_dataset_dir = os.path.join(
      'gs://conversationai-models/',
      getpass.getuser(),
      'tfrecords',
      'performance_dataset_dir')

  dataset = Dataset(input_fn, performance_dataset_dir)
  random.seed(2018) # Need to set seed before loading data to be able to reload same data in the future
  dataset.load_data(dataset_size, random_filter_keep_rate=0.5)
  dataset.add_model_prediction_to_data(model, recompute_predictions=True, class_names=class_names)
  scored_test_df = dataset.show_data()
  scored_test_df.to_csv(tf.gfile.Open(output_path, 'w'), index = False)

if __name__ == "__main__":
  model_names = [name.strip() for name in FLAGS.model_names.split(',')]
  print(model_names)
  class_names = [name.strip() for name in FLAGS.class_names.split(',')]
  print(class_names)
  score_data(model_names,
             class_names,
             FLAGS.test_data,
             FLAGS.output_path,
             FLAGS.project_name,
             FLAGS.text_feature_name,
             FLAGS.sentence_key,
             FLAGS.prediction_name,
             FLAGS.dataset_size)