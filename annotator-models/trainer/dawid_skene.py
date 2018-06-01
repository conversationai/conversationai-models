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
import math
import numpy as np
import pandas as pd
import sys
import time

FLAGS = None
np.set_printoptions(precision=2)

def run(items, raters, classes, counts, label, pseudo_count, tol=1, max_iter=25,
        init='average'):
    """
    Run the Dawid-Skene estimator on response data

    Input:
      responses: a pandas DataFrame of ratings where each row is a rating from
                 some rater ('_worker_id') on some item ('_unit_id')
      tol: tolerance required for convergence of EM
      max_iter: maximum number of iterations of EM
    """

    # initialize
    iteration = 0
    converged = False
    old_class_marginals = None
    old_error_rates = None

    # item_classes is a matrix of estimates of true item classes of size
    # [items, classes]
    item_classes = initialize(counts)
    [nItems, nRaters, nClasses] = np.shape(counts)

    logging.info('Iter\tlog-likelihood\tdelta-CM\tdelta-Y_hat')

    while not converged:
        iteration += 1
        start_iter = time.time()

        # M-step - updated error rates and class marginals given new
        #          distribution over true item classes
        old_item_classes = item_classes

        (class_marginals, error_rates) = m_step(counts, item_classes, pseudo_count)

        # E-step - calculate expected item classes given error rates and
        #          class marginals
        item_classes = e_step_verbose(counts, class_marginals, error_rates)

        # check likelihood
        log_L = calc_likelihood(counts, class_marginals, error_rates)

        # calculate the number of seconds the last iteration took
        iter_time = time.time() - start_iter

        # check for convergence
        if old_class_marginals is not None:
            class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
            item_class_diff = np.sum(np.abs(item_classes - old_item_classes))

            logging.info('{0}\t{1:.1f}\t{2:.4f}\t\t{3:.2f}\t({4:3.2f} secs)'.format(
                iteration, log_L, class_marginals_diff, item_class_diff, iter_time))

            if (class_marginals_diff < tol and item_class_diff < tol) \
               or iteration > max_iter:
                converged = True
        else:
            logging.info('{0}\t{1:.1f}'.format(iteration, log_L))

        # update current values
        old_class_marginals = class_marginals
        old_error_rates = error_rates

    return class_marginals, error_rates, item_classes

def load_data(path, unit_id, worker_id, label):
    logging.info('Loading data from {0}'.format(path))

    with tf.gfile.Open(path, 'rb') as fileobj:
      df =  pd.read_csv(fileobj, encoding='utf-8')

    # only keep necessary columns
    df = df[[unit_id, worker_id, label]]
    return df

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

def m_step(counts, item_classes, psuedo_count):
    """
    Get estimates for the prior class probabilities (p_j) and the error
    rates (pi_jkl) using MLE with current estimates of true item classes
    See equations 2.3 and 2.4 in Dawid-Skene (1979)

    Input:
      counts: Array of how many times each rating was given by each rater
        for each item
      item_classes: Matrix of current assignments of items to classes
      psuedo_count: A pseudo count used to smooth the error rates. For each rater k
        and for each class i and class j, we pretend rater k has rated
        psuedo_count examples with class i when class j was the true class.

    Returns:
      p_j: class marginals [classes]
      pi_kjl: error rates - the probability of rater k giving
          response l for an item in class j [observers, classes, classes]
    """
    [nItems, nRaters, nClasses] = np.shape(counts)

    # compute class marginals
    class_marginals = np.sum(item_classes, axis=0)/float(nItems)

    # compute error rates for each rater, each predicted class
    # and each true class

    error_rates = np.matmul(counts.T, item_classes) + psuedo_count

    # reorder axes so its of size [nItems x nClasses x nClasses]
    error_rates = np.einsum('abc->bca', error_rates)

    # divide each row by the sum of the error rates over all observation classes
    sum_over_responses = np.sum(error_rates, axis=2)[:,:,None]

    # for cases where an annotator has never used a label, set their sum over
    # responses for that label to 1 to avoid nan when we divide. The result will
    # be error_rate[k, i, j] is 0 if annotator k never used label i.
    sum_over_responses[sum_over_responses==0] = 1

    error_rates = np.divide(error_rates, sum_over_responses)

    return (class_marginals, error_rates)

def m_step_verbose(counts, item_classes, psuedo_count):
    """
    This method is the verbose (i.e. not vectorized) version of the m_step.
    It is currently not used because the vectorized version is faster, but we
    leave it here for future debugging.

    Get estimates for the prior class probabilities (p_j) and the error
    rates (pi_jkl) using MLE with current estimates of true item classes
    See equations 2.3 and 2.4 in Dawid-Skene (1979)

    Input:
      counts: Array of how many times each rating was given by each rater
        for each item
      item_classes: Matrix of current assignments of items to classes
      psuedo_count: A pseudo count used to smooth the error rates. For each rater k
        and for each class i and class j, we pretend rater k has rated
        psuedo_count examples with class i when class j was the true class.

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
    error_rates = np.zeros([nRaters, nClasses, nClasses])
    for k in range(nRaters):
        for j in range(nClasses):
            for l in range(nClasses):
                error_rates[k, j, l] = np.dot(item_classes[:,j], counts[:,k,l]) \
                                       + psuedo_count

            # normalize by summing over all observation classes
            sum_over_responses = np.sum(error_rates[k,j,:])

            if sum_over_responses > 0:
                error_rates[k,j,:] = error_rates[k,j,:]/float(sum_over_responses)

    return (class_marginals, error_rates)

def e_step(counts_tiled, class_marginals, error_rates):
    """
    Determine the probability of each item belonging to each class,
    given current ML estimates of the parameters from the M-step
    See equation 2.5 in Dawid-Skene (1979)

    Inputs:
      counts_tiled: A matrix of how many times each rating was given
          by each rater for each item, repeated for each class to make matrix
          multiplication fasterr. Size: [nItems, nRaters, nClasses, nClasses]
      class_marginals: probability of a random item belonging to each class.
          Size: [nClasses]
      error_rates: probability of rater k assigning a item in class j
          to class l. Size [nRaters, nClasses, nClasses]

    Returns:
      item_classes: Soft assignments of items to classes
          [items x classes]
    """
    [nItems, _, nClasses, _] = np.shape(counts_tiled)

    error_rates_tiled = np.tile(error_rates, (nItems,1,1,1))
    power = np.power(error_rates_tiled, counts_tiled)

    # Note, multiplying over axis 1 and then 2 is substantially faster than
    # the equivalent np.prod(power, axis=(1,3)
    item_classes = class_marginals * np.prod(np.prod(power, axis=1), axis=2)

    # normalize error rates by dividing by the sum over all classes
    item_sum = np.sum(item_classes, axis=1, keepdims=True)
    item_classes = np.divide(item_classes, np.tile(item_sum, (1, nClasses)))

    return item_classes

def e_step_verbose(counts, class_marginals, error_rates):
    """
    This method is the verbose (i.e. not vectorized) version of
    the e_step. It is actually faster than the vectorized e_step
    function (16 seconds vs 25 seconds respectively on 10k ratings).

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
        for j in range(nClasses):
            estimate = class_marginals[j]
            estimate *= np.prod(np.power(error_rates[:,j,:], counts[i,:,:]))
            item_classes[i,j] = estimate

    # normalize error rates by dividing by the sum over all classes
    item_sum = np.sum(item_classes, axis=1, keepdims=True)
    item_classes = np.divide(item_classes, np.tile(item_sum, (1, nClasses)))

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
        weights = response_sums[p,:] / np.sum(response_sums[p,:],dtype=float)
        item_classes[p,np.random.choice(np.arange(nClasses), p=weights)] = 1

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

def parse_item_classes(df, label, item_classes, index_to_unit_id_map, index_to_y_map, unit_id, worker_id):
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
    ROUND_DEC = 8
    _, N_ClASSES = np.shape(item_classes)

    df_predictions = pd.DataFrame()

    # Add columns for predictions for each class
    col_names = []
    for k in range(N_ClASSES):
        # y is the original value of the class. When we train, we re-map
        # all the classes to 0,1,....K. But our data has classes like
        # -2,-1,0,1,2. In that case, of k is 0, then y would be -2
        y = index_to_y_map[k]
        col_name = '{0}_{1}'.format(LABEL_HAT, y)
        col_names.append(col_name)

        df_predictions[col_name] = [round(i[k], ROUND_DEC) for i in item_classes]

    # To get a prediction of the mean label, multiply our predictions with the
    # true y values.
    y_values = index_to_y_map.values()
    col_name = '{0}_hat_mean'.format(label)
    df_predictions[col_name] = np.dot(df_predictions[col_names], y_values)

    # Use the _unit_index to map to the original _unit_id
    df_predictions['_unit_index'] = range(len(item_classes))
    df_predictions[unit_id] = df_predictions['_unit_index']\
                                 .apply(lambda i: index_to_unit_id_map[i])

    # Calculate the y_mean from the original data and join on _unit_id
    # Add a column for the mean predictions
    df[label] = df[label].astype(float)
    mean_labels = df.groupby(unit_id, as_index=False)[label]\
                   .mean()\
                   .round(ROUND_DEC)\
                   .rename(index=int, columns={label: LABEL_MEAN})
    df_predictions = pd.merge(mean_labels, df_predictions, on=unit_id)

    # join with data that contains the item-level comment text
    comment_text_path = FLAGS.comment_text_path
    with tf.gfile.Open(comment_text_path, 'r') as fileobj:
        logging.info('Loading comment text data from {}'.format(comment_text_path))
        df_comments = pd.read_csv(fileobj)

        # drop duplicate comments
        df_comments = df_comments.drop_duplicates(subset=unit_id)

    df_predictions = df_predictions.merge(df_comments, on=unit_id)
    return df_predictions

def parse_error_rates(df, error_rates, index_to_worker_id_map, index_to_y_map, unit_id, worker_id):
    """
    Given the original data DataFrame, the predicted error_rates and the mappings
    between the indexes and ids, returns a DataFrame with the fields:

      * _worker_index: the 0,1,...nItems index
      * _worker_id: the original item ID
      * _error_rate_{k}_{k}: probability the worker would choose class k when
          the true class is k (for accurate workers, these numbers are high).
    """
    columns = [worker_id, '_worker_index']

    df_error_rates = pd.DataFrame()

    # add the integer _worker_index
    df_error_rates['_worker_index'] = index_to_worker_id_map.keys()

    # add the original _worker_id
    df_error_rates[worker_id] = [j for (i,j) in index_to_worker_id_map.items()]

    # add annotation counts for each worker
    worker_counts = df.groupby(
        by=worker_id, as_index=False)[unit_id]\
                      .count()\
                      .rename(index=int, columns={unit_id: 'n_annotations'})

    df_error_rates = pd.merge(df_error_rates, worker_counts, on=worker_id)

    # add the diagonal error rates, which are the per-class accuracy rates,
    # for each class k, we add a column for p(rater will pick k | item's true class is k)

    # y_label is the original y value in the data and y_index is the
    # integer we mapped it to, i.e. 0, 1, ..., |Y|
    for y_index, y_label in index_to_y_map.items():
        col_name = 'accuracy_rate_{0}'.format(y_label)
        df_error_rates[col_name] = [e[y_index, y_index] for e in error_rates]

    return df_error_rates

def main(FLAGS):
    logging.basicConfig(level=logging.INFO)

    # load data, each row is an annotation
    n_examples= FLAGS.n_examples
    label = FLAGS.label
    unit_id = FLAGS.unit_id_col
    worker_id = FLAGS.worker_id_col
    df = load_data(FLAGS.data_path, unit_id, worker_id, label)[0:n_examples]

    logging.info('Running on {0} examples for label {1}'.format(len(df), label))

    # convert rater, item and label IDs to integers starting at 0
    #
    #   * worker_id_to_index_map: _worker_id -> index
    #   * index_to_worker_id_map: index -> worker
    #   * unit_id_to_index_map: _unit_id -> index
    #   * index_to_unit_id_map: index -> _unit_id
    #   * y_to_index_map: label -> index
    #   * index_to_y_map: index -> label
    worker_id_to_index_map = {w: i for (i,w) in enumerate(df[worker_id].unique())}
    index_to_worker_id_map = {i: w for (w,i) in  worker_id_to_index_map.items()}
    unit_id_to_index_map = {w: i for (i,w) in enumerate(df[unit_id].unique())}
    index_to_unit_id_map = {i: w for (w,i) in  unit_id_to_index_map.items()}
    y_to_index_map = {w: i for (i,w) in enumerate(df[label].unique())}
    index_to_y_map = {i: w for (w,i) in  y_to_index_map.items()}

    # create list of unique raters, items and labels
    raters = list(df[worker_id].apply(lambda x: worker_id_to_index_map[x]))
    items = list(df[unit_id].apply(lambda x: unit_id_to_index_map[x]))
    y = list(df[label].apply(lambda x: y_to_index_map[x]))

    nClasses = len(df[label].unique())
    nItems = len(df[unit_id].unique())
    nRaters = len(df[worker_id].unique())
    counts = np.zeros([nItems, nRaters, nClasses])

    # convert responses to counts
    for i,item_index in enumerate(items):
        rater_index = raters[i]
        y_index = y[i]
        counts[item_index,rater_index,y_index] += 1

    raters_unique = index_to_worker_id_map.keys()
    items_unique = index_to_unit_id_map.keys()
    classes_unique = index_to_y_map.keys()

    logging.info('num items: {0}'.format(len(items_unique)))
    logging.info('num raters: {0}'.format(len(raters_unique)))
    logging.info('num classes: {0}'.format(len(classes_unique)))

    # run EM
    start = time.time()
    class_marginals, error_rates, item_classes = run(
        items_unique, raters_unique, classes_unique, counts, label,
        FLAGS.pseudo_count,tol=FLAGS.tolerance, max_iter=FLAGS.max_iter)
    end = time.time()
    logging.info("training time: {0:.4f} seconds".format(end - start))

    # join comment_text, old labels and new labels
    df_predictions = parse_item_classes(df, label, item_classes, index_to_unit_id_map, index_to_y_map, unit_id, worker_id)

    # join rater error_rates
    df_error_rates = parse_error_rates(df, error_rates, index_to_worker_id_map, index_to_y_map, unit_id, worker_id)

    # write predictions and error_rates out as CSV
    n = len(df)
    prediction_path = '{0}/predictions_{1}_{2}.csv'.format(FLAGS.job_dir, label, n)
    error_rates_path = '{0}/error_rates_{1}_{2}.csv'.format(FLAGS.job_dir, label, n)

    logging.info('Writing predictions to {}'.format(prediction_path))
    with tf.gfile.Open(prediction_path, 'w') as fileobj:
      df_predictions.to_csv(fileobj, index=False, encoding='utf-8')

    logging.info('Writing error rates to {}'.format(error_rates_path))
    with tf.gfile.Open(error_rates_path, 'w') as fileobj:
      df_error_rates.to_csv(fileobj, index=False, encoding='utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',
                        help='The path to data to run on, local or in Cloud Storage.')
    parser.add_argument('--comment-text-path',
                        help='The path to comment text, local or in  Cloud Storage.')
    parser.add_argument('--worker-id-col',
                        help='Column name of worker id.', default='_worker_id')
    parser.add_argument('--unit-id-col',
                        help='Column name of unit id.', default='_comment_id')
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
    parser.add_argument('--pseudo-count', help='The pseudo count to smooth error rates.',
                        type=float, default=1.0)
    parser.add_argument('--tolerance',
                        help='Stop training when variables change less than this value.',
                        type=int, default=1)

    FLAGS = parser.parse_args()

    print('FLAGS', FLAGS)

    main(FLAGS)
