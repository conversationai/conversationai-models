"""
Evaluates a model on data. Takes a path to a directory with a TensorFlow
SavedModel and a path to a CSV with data, and outputs the predictions of the
model on the data.

Example usage:

python trainer/eval.py  \
  --model_dir=saved_models/{timestamp}/ \
  --test_data=local_data/test.csv
  --n_examples=1000
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, '')
from trainer import wikidata

# Name of the input words feature
WORDS_FEATURE = 'words'

# TODO: infer this from the saved model
MAX_DOCUMENT_LENGTH = 500

def predict(data, saved_model):
    """
    Args:
      data: (n, m) array of features where n is the number of examples
            and m is the number of features.
      saved_model: a SavedModelPredictor with an input tensor 'input' that
                  expects as input a list of tf.train.Example's.
    Returns:
      scores: (n, num_classes) array of scores for each each class
    """
    model_inputs = []

    for i, d in enumerate(data):
        input = tf.train.Example(
            features=tf.train.Features(
                feature={
                    WORDS_FEATURE: tf.train.Feature(
                        int64_list=tf.train.Int64List(value=d)
                    )
                }
            )
        )

        model_inputs.append(input.SerializeToString())

    output_dict = saved_model({'inputs': model_inputs})
    scores = output_dict['scores']

    return scores

def load_model(session, model_dir):

    """Loads SavedModelPredictor model"""
    tf.saved_model.loader.load(
        session, [tf.saved_model.tag_constants.SERVING], model_dir)

    saved_model = tf.contrib.predictor.from_saved_model(model_dir)

    return saved_model

def main():

    with tf.Session(graph=tf.Graph()) as sess:
        saved_model = load_model(sess, FLAGS.model_dir)

    # load test data
    data = wikidata.WikiData(
        data_path=FLAGS.test_data,
        max_document_length=MAX_DOCUMENT_LENGTH,
        model_dir=FLAGS.model_dir,
        predict_mode=True)

    # evaluate model on data
    x_to_eval = data.x_test[0:FLAGS.n_examples]
    scores = predict(x_to_eval, saved_model)

    results = {
        'results': [
            {
                'score': float(scores[i][0]),
                'text': data.x_test_text[i]
            } for i in range(len(scores))]
    }

    # write results to model directory
    output_path = '{0}predictions_{1}.json'.format(FLAGS.model_dir, FLAGS.n_examples)
    f = open(output_path, 'w')
    f.write(json.dumps(results))

    tf.logging.info('Wrote predictions to {0}'.format(output_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', help='Path to data to evaluate on.', type=str)
    parser.add_argument('--model_dir', help='Path to directory with TF model.', type=str)
    parser.add_argument('--n_examples', help='Number of examples to evaluate.',
                        type=int, default=100)

    FLAGS, unparsed = parser.parse_known_args()

    main()
