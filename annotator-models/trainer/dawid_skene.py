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
import logging
import numpy as np
import pandas as pd
import sys
import time

np.set_printoptions(precision=2, suppress=True)

FLAGS = None

def run(items, raters, classes, counts, label, tol=0.1, max_iter=100, init='average'):
    """
    Function: dawid_skene()
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

    logging.info('Iter\tlog-likelihood\tdelta-CM\tdelta-ER\tdelta-Y_hat')

    # while not converged do:
    while not converged:
        iter += 1

        # M-step - updated error rates and class marginals given new
        #          distribution over true item classes
        old_item_classes=item_classes
        (class_marginals, error_rates) = m_step(counts, item_classes)

        # E-setp - calculate expected item classes given error rates and
        #          class marginals
        item_classes = e_step(counts, class_marginals, error_rates)

        # check likelihood
        log_L = calc_likelihood(counts, class_marginals, error_rates)

        # check for convergence
        if old_class_marginals is not None:
            class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
            error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))
            item_class_diff = np.sum(np.abs(item_classes - old_item_classes))

            logging.info('{0}\t{1:.1f}\t{2:.4f}\t{3:.4f}\t{4:.4f}'.format(
                iter, log_L, class_marginals_diff, error_rates_diff, item_class_diff))

            if (class_marginals_diff < tol and error_rates_diff < tol) or iter > max_iter:
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
    Function: responses_to_counts()
      Convert a matrix of annotations to count data

    Inputs:
      responses: dictionary of responses {patient:{observers:[responses]}}
      label: string of the toxicity type to use, e.g. 'toxic_score' or 'obscene'

    Return:
      patients: list of patients
      observers: list of observers
      classes: list of possible patient classes
      counts: 3d array of counts: [patients x observers x classes]
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

    return (unique_items, unique_raters, unique_classes, counts)


def initialize(counts):
    """
    Function: initialize()
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
    Function: m_step()
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
    error_rates = np.zeros([nRaters, nClasses, nClasses])
    for k in range(nRaters):
        for j in range(nClasses):
            for l in range(nClasses):

                error_rates[k, j, l] = np.dot(item_classes[:,j], counts[:,k,l])

            # normalize by summing over all observation classes
            sum_over_responses = np.sum(error_rates[k,j,:])
            if sum_over_responses > 0:
                error_rates[k,j,:] = error_rates[k,j,:]/float(sum_over_responses)

    return (class_marginals, error_rates)

def e_step(counts, class_marginals, error_rates):
    """
    Function: e_step()
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
        patient_sum = np.sum(item_classes[i,:])
        if patient_sum > 0:
            item_classes[i,:] = item_classes[i,:]/float(patient_sum)

    return item_classes

def calc_likelihood(counts, class_marginals, error_rates):
    """
    Function: calc_likelihood()
      Calculate the likelihood given the current parameter estimates
      This should go up monotonically as EM proceeds
      See equation 2.7 in Dawid-Skene (1979)

    Inputs:
      counts: Array of how many times each response was received
          by each observer from each patient
      class_marginals: probability of a random patient belonging to each class
      error_rates: probability of observer k assigning a patient in class j
          to class l [observers, classes, classes]
    Returns:
      Likelihood given current parameter estimates
    """
    [nItems, nRaters, nClasses] = np.shape(counts)
    log_L = 0.0

    for i in range(nItems):
        patient_likelihood = 0.0
        for j in range(nClasses):

            class_prior = class_marginals[j]
            patient_class_likelihood = np.prod(np.power(error_rates[:,j,:], counts[i,:,:]))
            patient_class_posterior = class_prior * patient_class_likelihood
            patient_likelihood += patient_class_posterior

        temp = log_L + np.log(patient_likelihood)

        if np.isnan(temp) or np.isinf(temp):
            logging.info("{0}, {1}, {2}".format(i, log_L, np.log(patient_likelihood), temp))
            sys.exit()

        log_L = temp

    return log_L

def random_initialization(counts):
    """
    Function: random_initialization()
      Alternative initialization # 1
      Similar to initialize() above, except choose one initial class for each
      patient, weighted in proportion to the counts
    Input:
      counts: counts of the number of times each response was received
          by each observer from each patient: [patients x observers x classes]
    Returns:
      item_classes: matrix of estimates of true patient classes:
          [patients x responses]
    """
    [nItems, nRaters, nClasses] = np.shape(counts)

    response_sums = np.sum(counts,1)

    # create an empty array
    item_classes = np.zeros([nItems, nClasses])

    # for each patient, choose a random initial class, weighted in proportion
    # to the counts from all observers
    for p in range(nItems):
        average = response_sums[p,:] / np.sum(response_sums[p,:],dtype=float)
        item_classes[p,np.random.choice(np.arange(nClasses), p=average)] = 1

    return item_classes

def majority_voting(counts):
    """
    Function: majority_voting()
      Alternative initialization # 2
      An alternative way to initialize assignment of patients to classes
      i.e Get initial estimates for the true patient classes using majority voting
      This is not in the original paper, but could be considered

    Input:
      counts: Counts of the number of times each response was received
          by each rater from each patient: [items x raters x classes]
    Returns:
      item_classes: matrix of initial estimates of true item classes:
          [items x responses]
    """
    [nItems, nRaters, nClasses] = np.shape(counts)
    # sum over observers
    response_sums = np.sum(counts,1)

    # create an empty array
    item_classes = np.zeros([nItems, nClasses])

    # take the most frequent class for each patient
    for p in range(nItems):
        indices = np.argwhere(response_sums[p,:] == np.max(response_sums[p,:]))
        # in the case of ties, take the lowest valued label (could be randomized)
        item_classes[p, np.min(indices)] = 1

    return item_classes

def main(FLAGS):
    # configure logging
    logging.basicConfig(level=logging.INFO)

    # load data, each row is an annotation
    n_examples= FLAGS.n_examples
    label = FLAGS.label

    df = load_data(FLAGS.data_path)[0:n_examples]

    logging.info('Running on {0} examples for label {1}'.format(len(df), label))

    # convert responses to counts
    (items, raters, classes, counts) = responses_to_counts(df, label)
    logging.info('num items: {0}'.format(len(items)))
    logging.info('num raters: {0}'.format(len(raters)))
    logging.info('num classes: {0}'.format(len(classes)))

    # run EM
    start = time.time()
    class_marginals, error_rates, item_classes = run(
        items, raters, classes, counts, label=label, max_iter=50, tol=.1)
    end = time.time()
    logging.info("training time: {0:.4f} seconds".format(end - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',
                        help='The path to data to run on, local or in Cloud Storage.')
    parser.add_argument('--n_examples',
                        help='The number of annotations to use.', default=10000000)
    parser.add_argument('--label', help='The label to train on, e.g. "obscene" or "threat"',
                        default='obscene')
    parser.add_argument("--job-dir", type=str, default="",
                        help="The directory where the job is staged")

    FLAGS = parser.parse_args()

    main(FLAGS)
