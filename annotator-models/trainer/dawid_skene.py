"""
Description:
Given unreliable ratings of items classes by multiple raters, determine the most
likely true class for each item, class marginals, and  individual error rates
for each rater, using Expectation Maximization

References:
( Dawid and Skene (1979). Maximum Likelihood Estimation of Observer
Error-Rates Using the EM Algorithm. Journal of the Royal Statistical Society.
Series C (Applied Statistics), Vol. 28, No. 1, pp. 20-28.
"""

import argparse
import tensorflow as tf
from scipy import stats
import logging
import numpy as np
import pandas as pd
import sys
import time

np.set_printoptions(precision=2, suppress=True)

FLAGS = None

def run(items, raters, classes, counts, label, tol=0.1, max_iter=100, init='average'):
    """
    Run the Dawid-Skene estimator on response data

    Input:
      responses: a pandas DataFrame of ratings where each row is a rating from
                 some rater ('_worker_id') on some item ('_unit_id')
      tol: tolerance required for convergence of EM
      max_iter: maximum number of iterations of EM
    """

    # initialize
    iter = 0
    converged = False
    old_class_marginals = None
    old_error_rates = None

    # item_classes is a matrix of estimates of true item classes of size
    # [items, classes]
    item_classes = initialize(counts)
    # item_classes = random_initialization(counts)

    logging.info('Iter\tlog-likelihood\tdelta-CM\tdelta-Y_hat')

    # while not converged do:
    while not converged:
        iter += 1

        # M-step - updated error rates and class marginals given new
        #          distribution over true item classes
        old_item_classes = item_classes
        (class_marginals, error_rates) = m_step(counts, item_classes)

        # E-setp - calculate expected item classes given error rates and
        #          class marginals
        item_classes = e_step(counts, class_marginals, error_rates)

        # check likelihood
        log_L = calc_likelihood(counts, class_marginals, error_rates)

        # check for convergence
        if old_class_marginals is not None:
            class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
            item_class_diff = np.sum(np.abs(item_classes - old_item_classes))

            logging.info('{0}\t{1:.1f}\t{2:.4f}\t{3:.4f}'.format(
                iter, log_L, class_marginals_diff, item_class_diff))


            if (class_marginals_diff < tol and item_class_diff < tol) or iter > max_iter:
                converged = True
        else:
            logging.info('{0}\t{1:.1f}'.format(iter, log_L))

        # update current values
        old_class_marginals = class_marginals
        old_error_rates = error_rates

    return class_marginals, error_rates, item_classes


def load_data(path):
    logging.info('Loading data from {0}'.format(path))

    with tf.gfile.Open(path, 'rb') as fileobj:
      df =  pd.read_csv(fileobj, encoding='utf-8')

    # Remove all rows with nan values
    df = df[df.isnull().any(axis=1) == False]
    return df

def responses_to_counts(df, label):
    """
    Convert a matrix of annotations to count data

    Inputs:
      df: pandas DataFrame that includes columns '_worker_id', '_unit_id' and label
      label: string of the toxicity type to use, e.g. 'toxic_score' or 'obscene'

    Return:
      items: list of items
      raters: list of raters
      classes: list of possible item classes
      counts: 3d array of counts: [items x raters x classes]
    """
    # _worker_id -> index and index -> worker
    worker_id_to_index_map = {w: i for (i,w) in enumerate(df["_worker_id"].unique())}
    index_to_worker_id_map = {i: w for (w,i) in  worker_id_to_index_map.items()}

    # _unit_id -> index and index -> _unit_id
    unit_id_to_index_map = {w: i for (i,w) in enumerate(df['_unit_id'].unique())}
    index_to_unit_id_map = {i: w for (w,i) in  unit_id_to_index_map.items()}

    # label -> index and index -> label
    y_to_index_map = {w: i for (i,w) in enumerate(df[label].unique())}
    index_to_y_map = {i: w for (w,i) in  y_to_index_map.items()}

    raters = list(df['_worker_id'].apply(lambda x: worker_id_to_index_map[x]))
    items = list(df['_unit_id'].apply(lambda x: unit_id_to_index_map[x]))

    y = list(df[label].apply(lambda x: y_to_index_map[x]))

    nClasses = len(df[label].unique())
    nItems = len(df['_unit_id'].unique())
    nRaters = len(df['_worker_id'].unique())
    counts = np.zeros([nItems, nRaters, nClasses])

    # convert responses to counts
    for i,item_index in enumerate(items):
        rater_index = raters[i]
        y_index = y[i]
        counts[item_index,rater_index,y_index] += 1

    unique_raters = index_to_worker_id_map.keys()
    unique_items = index_to_unit_id_map.keys()
    unique_classes = index_to_y_map.keys()

    return unique_items, unique_raters, unique_classes, counts, index_to_unit_id_map


def initialize(counts):
    """
    Get initial estimates for the true item classes using counts
    see equation 3.1 in Dawid-Skene (1979)

    Input:
      counts: counts of the number of times each response was given
          by each rater for each item: [items x raters x classes]. Note
          in the crowd rating example, counts will be a 0/1 matrix.

    Returns:
      item_classes: matrix of estimates of true item classes:
          [items x responses]
    """
    [nItems, nRaters, nClasses] = np.shape(counts)

    # sum over raters
    response_sums = np.sum(counts,1)

    # create an empty array
    item_classes = np.zeros([nItems, nClasses])

    # for each item, take the average number of ratings in each class
    for p in range(nItems):
        item_classes[p,:] = response_sums[p,:] / np.sum(response_sums[p,:],dtype=float)

    return item_classes

def m_step(counts, item_classes):
    """
    Get estimates for the prior class probabilities (p_j) and the error
    rates (pi_jkl) using MLE with current estimates of true item classes
    See equations 2.3 and 2.4 in Dawid-Skene (1979)

    Input:
      counts: Array of how many times each rating was given by each rater
        for each item
      item_classes: Matrix of current assignments of items to classes

    Returns:
      p_j: class marginals [classes]
      pi_kjl: error rates - the probability of rater k giving
          response l for an item in class j [observers, classes, classes]
    """
    [nItems, nRaters, nClasses] = np.shape(counts)

    # compute class marginals
    class_marginals = np.sum(item_classes, 0)/float(nItems)

    # compute error rates for each rater, each predicted class
    # and each true class
    error_rates_1 = np.matmul(counts.T, item_classes)

    # Re-order axes so its of size [nItems x nClasses x nClasses]
    error_rates_1 = np.einsum('abc->bca', error_rates_1)

    # Divide each row by the sum of the error rates over all observation classes
    sum_over_responses = np.sum(error_rates_1, axis=2)[:,:,None]
    error_rates_1 = np.divide(error_rates_1, sum_over_responses, where=sum_over_responses!=0)
    #error_rates_1 = np.round(error_rates_1, 5)
    # tol = 1e-8
    # error_rates_1[np.abs(error_rates_1) < tol] = 0.0

    # error_rates = np.zeros([nRaters, nClasses, nClasses])

    # for k in range(nRaters):
    #     # [nClasses, nClasses] = [nClasses x nItems], [nItems, nClasses]
    #     error_rates[k, :, :] = np.matmul(item_classes.T, counts[:,k,:])
    #     sum_over_responses = np.sum(error_rates[k,:,:], axis=1)[:,None]

    #     # Divide each row by the sum over all observation classes
    #     error_rates[k,:,:] = np.divide(
    #         error_rates[k,:,:], sum_over_responses, where=sum_over_responses!=0)

    return (class_marginals, error_rates_1)

def e_step(counts, class_marginals, error_rates):
    """
    Determine the probability of each item belonging to each class,
    given current ML estimates of the parameters from the M-step
    See equation 2.5 in Dawid-Skene (1979)

    Inputs:
      counts: Array of how many times each rating was given
          by each rater for each item
      class_marginals: probability of a random item belonging to each class
      error_rates: probability of rater k assigning a item in class j
          to class l [raters, classes, classes]

    Returns:
      item_classes: Soft assignments of items to classes
          [items x classes]
    """
    [nItems, nRaters, nClasses] = np.shape(counts)

    item_classes = np.zeros([nItems, nClasses])

    for i in range(nItems):

        # counts_i = np.stack([counts[i,:,:],counts[i,:,:]],axis=1)
        # estimate_1 = np.prod(np.power(error_rates,counts_i), axis=(0,2))
        # estimate_1 = class_marginals *  estimate_1
        # item_classes[i,:] = estimate_1

        for j in range(nClasses):
            estimate = class_marginals[j]
            estimate *= np.prod(np.power(error_rates[:,j,:], counts[i,:,:]))

            item_classes[i,j] = estimate

        # normalize error rates by dividing by the sum over all classes
        item_sum = np.sum(item_classes[i,:])
        if item_sum > 0:
            item_classes[i,:] = item_classes[i,:]/float(item_sum)

    return item_classes

def calc_likelihood(counts, class_marginals, error_rates):
    """
    Calculate the likelihood given the current parameter estimates
    This should go up monotonically as EM proceeds
    See equation 2.7 in Dawid-Skene (1979)

    Inputs:
      counts: Array of how many times each response was received
          by each rater from each item
      class_marginals: probability of a random item belonging to each class
      error_rates: probability of rater k assigning a item in class j
          to class l [raters, classes, classes]

    Returns:
      Likelihood given current parameter estimates
    """
    [nItems, nRaters, nClasses] = np.shape(counts)
    log_L = 0.0

    for i in range(nItems):
        item_likelihood = 0.0
        for j in range(nClasses):

            class_prior = class_marginals[j]
            item_class_likelihood = np.prod(np.power(error_rates[:,j,:], counts[i,:,:]))
            item_class_posterior = class_prior * item_class_likelihood
            item_likelihood += item_class_posterior

        temp = log_L + np.log(item_likelihood)

        if np.isnan(temp) or np.isinf(temp):
            logging.info("{0}, {1}, {2}".format(i, log_L, np.log(item_likelihood), temp))
            sys.exit()

        log_L = temp

    return log_L

def random_initialization(counts):
    """
    Similar to initialize() above, except choose one initial class for each
    item, weighted in proportion to the counts.

    Input:
      counts: counts of the number of times each response was received
          by each rater from each item: [items x raters x classes]

    Returns:
      item_classes: matrix of estimates of true item classes:
          [items x responses]
    """
    [nItems, nRaters, nClasses] = np.shape(counts)

    response_sums = np.sum(counts,1)

    # create an empty array
    item_classes = np.zeros([nItems, nClasses])

    # for each item, choose a random initial class, weighted in proportion
    # to the counts from all raters
    for p in range(nItems):
        average = response_sums[p,:] / np.sum(response_sums[p,:],dtype=float)
        item_classes[p,np.random.choice(np.arange(nClasses), p=average)] = 1

    return item_classes

def majority_voting(counts):
    """
      An alternative way to initialize assignment of items to classes
      i.e Get initial estimates for the true item classes using majority voting

    Input:
      counts: Counts of the number of times each response was received
          by each rater from each item: [items x raters x classes]
    Returns:
      item_classes: matrix of initial estimates of true item classes:
          [items x responses]
    """
    [nItems, nRaters, nClasses] = np.shape(counts)
    # sum over observers
    response_sums = np.sum(counts,1)

    # create an empty array
    item_classes = np.zeros([nItems, nClasses])

    # take the most frequent class for each item
    for p in range(nItems):
        indices = np.argwhere(response_sums[p,:] == np.max(response_sums[p,:]))
        # in the case of ties, take the lowest valued label (could be randomized)
        item_classes[p, np.min(indices)] = 1

    return item_classes

def parse_results(df, label, item_classes, index_to_unit_id_map):
    """
    Given the original data df, the predicted item_classes, and
    the data mappings, returns a DataFrame with the fields:
      * _unit_index: the 0,1,...nItems index
      * _unit_id: the original item ID
      * {LABEL}_hat: the predicted probability of the item being labeled 1 as
               learned from the Dawid-Skene algorithm
      * {LABEL}_mean: the mean of the original ratings
    """
    LABEL_HAT = '{}_hat'.format(label)
    LABEL_MEAN = '{}_mean'.format(label)
    ROUND_DEC = 4

    df_predictions = pd.DataFrame()
    df_predictions[LABEL_HAT] = [round(i[1], ROUND_DEC) for i in item_classes]
    df_predictions['_unit_index'] = range(len(item_classes))

    # Use the _unit_index to map to the original _unit_id
    df_predictions['_unit_id'] = df_predictions['_unit_index']\
                                 .apply(lambda i: int(index_to_unit_id_map[i]))

    # Calculate the y_mean from the original data and join on _unit_id
    df[label] = df[label].astype(float)
    mean_labels = df.groupby('_unit_id', as_index=False)[label]\
                   .mean()\
                   .round(ROUND_DEC)\
                   .rename(index=int, columns={label: LABEL_MEAN})
    df_predictions = pd.merge(mean_labels, df_predictions, on='_unit_id')

    return df_predictions

def main(FLAGS):
    logging.basicConfig(level=logging.INFO)

    # load data, each row is an annotation
    n_examples= FLAGS.n_examples
    label = FLAGS.label
    df = load_data(FLAGS.data_path)[0:n_examples]

    logging.info('Running on {0} examples for label {1}'.format(len(df), label))

    # convert responses to counts
    items, raters, classes, counts, index_to_unit_id_map = responses_to_counts(df, label)

    logging.info('num items: {0}'.format(len(items)))
    logging.info('num raters: {0}'.format(len(raters)))
    logging.info('num classes: {0}'.format(len(classes)))

    # run EM
    start = time.time()
    class_marginals, error_rates, item_classes = run(
        items, raters, classes, counts, label=label, max_iter=18, tol=.1)
    end = time.time()
    logging.info("training time: {0:.4f} seconds".format(end - start))

    df_predictions = parse_results(df, label, item_classes, index_to_unit_id_map)

    # save error_rates, item_classes and class_marginals
    # TK

    # save predictions as CSV to Cloud Storage
    n = len(df)
    prediction_path = '{0}/predictions_{1}_{2}.csv'.format(FLAGS.job_dir, label, n)
    with tf.gfile.Open(prediction_path, 'w') as fileobj:
      df_predictions.to_csv(fileobj, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',
                        help='The path to data to run on, local or in Cloud Storage.')
    parser.add_argument('--n_examples',
                        help='The number of annotations to use.', default=10000000,
                        type=int)
    parser.add_argument('--label', help='The label to train on, e.g. "obscene" or "threat"',
                        default='obscene')
    parser.add_argument("--job-dir", type=str, default="",
                        help="The directory where the job is staged.")
    parser.add_argument('--max-iter',
                        help='The max number of iteration to run.', type=int,
                        default=25)

    FLAGS = parser.parse_args()

    main(FLAGS)
