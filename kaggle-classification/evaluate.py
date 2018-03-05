"""
Output:
  * writes predictions on heldout test data to TEST_OUT_PATH
  * writes predictions on unlabled predict data to PREDICT_OUT_PATH
"""

# # Output Params
# TEST_OUT_PATH = 'test_out.csv' # Where to write results on heldout data
# PREDICT_OUT_PATH = 'predict_out.csv' # Where to write results on unlabled data

# # If specified, predict on unlabeled data
# if FLAGS.predict_data is None:
#   return

# data_unlabeled = WikiData(FLAGS.predict_data).data

# tf.logging.info('Generating predictions for {0} unlabeled examples in {1}'
#                 .format(len(data_unlabeled), FLAGS.predict_data))

# x_unlabeled = np.array(list(
#   vocab_processor.fit(data_unlabeled['comment_text'])))

# unlabled_input_fn = tf.estimator.inputs.numpy_input_fn(
#   x={WORDS_FEATURE: x_unlabeled},
#   num_epochs=1,
#   shuffle=False)

# predicted_unlabeled = classifier.predict(input_fn=unlabled_input_fn)
# unlabeled_out = pd.DataFrame(
#   [(p['classes'], p['probs'][1]) for p in predicted_unlabeled],
#   columns=['y_pred', 'prob']
# )
# unlabeled_out['comment_text'] = data_unlabeled['comment_text']

# # Write out predictions and probabilities for unlabled "predict" data
# tf.logging.info("Writing predictions to {}".format(PREDICT_OUT_PATH))
# unlabeled_out.to_csv(PREDICT_OUT_PATH)

# if __name__ == '__main__':

#   parser = argparse.ArgumentParser()
#   parser.add_argument(
#       '--verbose', help='Run in verbose mode.', action='store_true')
#   parser.add_argument(
#       "--train_data", type=str, default="", help="Path to the training data.")
#   parser.add_argument(
#       "--predict_data", type=str, default="", help="Path to the prediction data.")
#   parser.add_argument(
#       "--y_class", type=str, default="toxic",
#     help="Class to train model against, one of {}".format(Y_CLASSES))
#   parser.add_argument(
#       "--model", type=str, default="bag_of_words",
#     help="The model to train, one of {}".format(MODEL_LIST))

#   FLAGS, unparsed = parser.parse_known_args()

#   main()


#     #   test_out = pd.DataFrame(
#     #   [(p['classes'], p['probs'][1]) for p in predicted_test],
#     #   columns=['y_predicted', 'prob']
#     # )
#     # test_out['comment_text'] = data.x_train_text
#     # test_out['y_true'] = data.y_test

#     # # Write out predictions and probabilities for test data
#     # tf.logging.info("Writing test predictions to {}".format(TEST_OUT_PATH))
#     # test_out.to_csv(TEST_OUT_PATH)

#         with tf.Session(graph=tf.Graph()) as sess:
#       import pdb; pdb.set_trace()

#       tf.saved_model.loader.load(sess, [tag_constants.TRAINING], dir_path)
