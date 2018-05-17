# Evaluates a model on a gold data set.
#
# Example usage:
# ./eval_on_gold \
#    --data_path=gold.csv \
#    --output_filename=test_data_scored.csv \
#    --model_dir=models/my_model/
#
# Example gold data:
#
# comment_text,label,gold,_unit_id
# "..and kill things",threat,0,15190
# "Die!",threat,1,15193
# "FU",obscene,1,15089
# "Have an apple",flirtation,0,15789
#
# where the 'label' field corresponds with one of the heads of the model.
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from keras_trainer.model import ModelRunner

def calculate_auc(df_gold_scored):
    # HACK: the comet-ml dependency in keras-trainer throws
    # an error when you try to import sklearn.metrics before
    # you import comet-ml.
    from sklearn import metrics

    # only examples with 0/1 gold labels
    df = df_gold_scored[df_gold_scored['gold'].isin([0,1])]

    y_score = [row[row['label']]  for _, row in df.iterrows()]
    y_true = np.array(df['gold'])

    auc = metrics.roc_auc_score(y_true, y_score)

    return auc

def main(FLAGS):
    # load gold data
    with tf.gfile.Open(FLAGS.gold_path, 'rb') as f:
        df_gold = pd.read_csv(f, encoding='utf-8')

    # load model
    model = ModelRunner(
        job_dir=FLAGS.job_dir,
        embeddings_path=FLAGS.embeddings_path,
        log_path='',
        hparams=None, # use the defualt params
        labels=FLAGS.labels)

    print('Running data through model')
    scores = model.predict(df_gold['comment_text'])
    df_scores = pd.DataFrame(scores, columns=model.labels)
    df_gold_scored = pd.concat([df_scores, df_gold], axis=1)

    # evaluate accuracy

    # TODO: filter out bad annotations by checking if annotation
    # is a float between 0 and 1
    df_gold_scored['bad_annotation'] = df_gold_scored['gold']\
                                       .apply(lambda x: x.strip() == '')
    df_gold_scored = df_gold_scored[df_gold_scored['bad_annotation'] == False]

    df_gold_scored['gold'] = df_gold_scored['gold'].astype('float64')
    df_gold_scored['abs_gold_diff'] = [
        np.abs(row[row['label']]  - row['gold'])
        for _, row in df_gold_scored.iterrows()]

    # TODO: calculate more metrics
    avg_diff = df_gold_scored['abs_gold_diff'].mean()
    print('Average Difference:', avg_diff)

    auc = calculate_auc(df_gold_scored)
    print('AUC', auc)

    # write out evaluation results
    evals = {
        'auc': auc,
        'avg_diff': avg_diff,
        'gold_path': FLAGS.gold_path
    }
    eval_path = '{0}/gold_eval.json'.format(FLAGS.job_dir)

    with tf.gfile.Open(eval_path, 'w') as f:
        f.write(json.dumps(evals))

    # write out scored data
    print('Writing scored data to {0}'.format(FLAGS.output_path))
    with tf.gfile.Open(FLAGS.output_path, 'w') as f:
        df_gold_scored.to_csv(f, encoding='utf-8', index=False)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--gold-path', type=str, default='local_data/gold_toxicity_subtypes.csv',
      help='Path to gold test data')
  parser.add_argument(
      '--output-path', type=str, default='gold_scored.csv',
      help='Name of the file to write results to')
  parser.add_argument(
      '--job-dir', type=str,
      help='Path job directory for the model')
  # Note: the list of labels MUST be in the same order as the list of
  # labels used to specify the model. The output heads do not have
  # labels, they just have an ordering. TODO: fix this
  parser.add_argument(
      '--labels', type=str,
      default='threat,flirtation,identity_hate,insult,obscene,sexual_explicit,frac_very_neg,frac_neg',
      help='A comma separated list of labels to predict.')
  parser.add_argument(
      '--embeddings_path', type=str,
      default='local_data/glove.6B/glove.6B.100d.txt',
      help='Path to the embeddings.')

  FLAGS = parser.parse_args()

  main(FLAGS)
