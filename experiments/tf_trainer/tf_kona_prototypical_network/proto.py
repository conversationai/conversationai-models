import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import sys
import datetime
import collections

tf.app.flags.DEFINE_string(
    "train_file",
    "gs://conversationai-models/resources/transfer_learning_data/many_communities_40_per_8_shot/train_cleaned_text.csv",
    "CSV file containing the training data. Expects columns: domain, label, support_or_query"
)
tf.app.flags.DEFINE_string(
    "validation_file",
    "gs://conversationai-models/resources/transfer_learning_data/many_communities_40_per_8_shot/validation_cleaned_text.csv",
    "CSV file containing the validation data. Expects columns: domain, label, support_or_query"
)
tf.app.flags.DEFINE_string(
    "test_file",
    "gs://conversationai-models/resources/transfer_learning_data/many_communities_40_per_8_shot/test_cleaned_text.csv",
    "CSV file containing the test data. Expects columns: domain, label, support_or_query"
)
tf.app.flags.DEFINE_boolean(
    "test_mode", False,
    "If true then no training occurs and it prints out metrics on the test set."
)
tf.app.flags.DEFINE_string("model_dir", "", "The model directory in GCS.")
tf.app.flags.DEFINE_string(
    "encoding_layers", "256,128",
    "Comma delimited integers representing the number of units for each dense layer."
)

FLAGS = tf.app.flags.FLAGS


def distance(embeddings, prototype):
  return tf.map_fn(tf.norm, embeddings - prototype)


def neg_distance(embs, proto):
  return -distance(embs, proto)


def calculate_logits(embeddings, positive_prototype, negative_prototype):
  negative_logits = neg_distance(embeddings, negative_prototype)
  positive_logits = neg_distance(embeddings, positive_prototype)
  return tf.stack([negative_logits, positive_logits], axis=1)


def prepare_dataset(data):
  data["text"] = data.text.fillna("")
  domains = data.domain.unique()

  positive_supports = []
  positive_queries = []
  negative_supports = []
  negative_queries = []

  for domain in domains:
    domain_data = data[data["domain"] == domain]
    positive = domain_data[domain_data["label"] == 1]
    negative = domain_data[domain_data["label"] == 0]
    positive_support = positive[positive["support_or_query"] == "support"].text
    positive_query = positive[positive["support_or_query"] == "query"].text
    negative_support = negative[negative["support_or_query"] == "support"].text
    negative_query = negative[negative["support_or_query"] == "query"].text

    positive_supports.append(positive_support)
    positive_queries.append(positive_query)
    negative_supports.append(negative_support)
    negative_queries.append(negative_query)

  return tf.data.Dataset.from_tensor_slices({
      "positive_supports": np.array(positive_supports),
      "negative_supports": np.array(negative_supports),
      "positive_queries": np.array(positive_queries),
      "negative_queries": np.array(negative_queries)
  })


def encoder(dense_config, output_types, output_shapes):
  """Tensorflow graph for getting prototypes and embeddings.

  It contains a placeholder for a tensorflow Iterator handle whose elements
  are a dict containing negative_supports, positive_supports, negative_queries,
  and positive_queries. All of these are lists of strings.

  Args:
    dense_config: A list of integers that configure the dense layers.

  Returns:
    A tuple of logits, the first representing those from the negative query set
    and the second from the positive query set.
  """

  if not dense_config:
    raise ValueError("encoder must be called with a non empty dense_config")

  embed = hub.Module(
      "https://tfhub.dev/google/universal-sentence-encoder-large/3")
  dense_layers = [
      tf.keras.layers.Dense(units, activation=tf.nn.relu)
      for units in dense_config
  ]
  last_layer = tf.keras.layers.Dense(dense_config[-1], activation=None)

  def get_embeddings(texts):
    result = embed(texts)
    for dense_layer in dense_layers:
      result = dense_layer(result)
    return last_layer(result)

  get_prototype = lambda texts: tf.reduce_mean(get_embeddings(texts), 0)

  handle = tf.placeholder(tf.string, shape=[])
  iterator = tf.data.Iterator.from_string_handle(handle, output_types,
                                                 output_shapes)
  episode_batch = iterator.get_next()

  with tf.variable_scope("negative_prototype"):
    negative_prototype = get_prototype(episode_batch["negative_supports"])
  with tf.variable_scope("positive_prototype"):
    positive_prototype = get_prototype(episode_batch["positive_supports"])
  with tf.variable_scope("negative_embeddings"):
    negative_embeddings = get_embeddings(episode_batch["negative_queries"])
  with tf.variable_scope("positive_embeddings"):
    positive_embeddings = get_embeddings(episode_batch["positive_queries"])

  negative_logits = calculate_logits(negative_embeddings, positive_prototype,
                                     negative_prototype)
  positive_logits = calculate_logits(positive_embeddings, positive_prototype,
                                     negative_prototype)

  return handle, negative_logits, positive_logits


def train_operation(negative_logits, positive_logits):
  negative_loss = tf.losses.softmax_cross_entropy(
      tf.broadcast_to(tf.one_hot(0, 2), tf.shape(negative_logits)),
      negative_logits)
  positive_loss = tf.losses.softmax_cross_entropy(
      tf.broadcast_to(tf.one_hot(1, 2), tf.shape(positive_logits)),
      positive_logits)
  loss = negative_loss + positive_loss

  optimizer = tf.train.AdamOptimizer(0.001)
  train = optimizer.minimize(loss)
  return (train, loss)


def predictions_and_metrics(negative_logits, positive_logits):
  predict = lambda logits: tf.argmax(logits, axis=1)

  negative_predictions = predict(negative_logits)
  negative_labels = tf.fill(tf.shape(negative_predictions), 0)
  positive_predictions = predict(positive_logits)
  positive_labels = tf.fill(tf.shape(positive_predictions), 1)

  probability = tf.nn.softmax(
      tf.concat([negative_logits, positive_logits], -2), axis=-1)
  labels = tf.concat([negative_labels, positive_labels], -1)
  predictions = tf.concat([negative_predictions, positive_predictions], -1)

  acc_op, update_acc_op = tf.metrics.accuracy(labels, predictions)
  auc_op, update_auc_op = tf.metrics.auc(labels,
                                         tf.gather(probability, 1, axis=-1))
  return (predictions, acc_op, auc_op, update_acc_op, update_auc_op)


if FLAGS.model_dir:
  model_dir = FLAGS.model_dir
else:
  st = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  model_dir = "gs://conversationai-models/jjtan/transfer_learning/model/" + st
print("Model dir: " + model_dir)
save_path = model_dir + "/save/model.ckpt"
metadata_path = model_dir + "/meta.txt"

with tf.gfile.Open(metadata_path, "w") as f:
  f.write("Encoding Layers: " + FLAGS.encoding_layers + "\n")

output_types = {
    "negative_queries": tf.string,
    "negative_supports": tf.string,
    "positive_queries": tf.string,
    "positive_supports": tf.string
}
output_shapes = {
    "negative_queries": tf.TensorShape([tf.Dimension(12)]),
    "negative_supports": tf.TensorShape([tf.Dimension(8)]),
    "positive_queries": tf.TensorShape([tf.Dimension(12)]),
    "positive_supports": tf.TensorShape([tf.Dimension(8)])
}

with tf.variable_scope("encoder"):
  encoding_units = [int(units) for units in FLAGS.encoding_layers.split(",")]
  handle, negative_logits, positive_logits = encoder(encoding_units,
                                                     output_types,
                                                     output_shapes)

if FLAGS.test_mode:
  print("In TEST mode.")
  with tf.gfile.Open(FLAGS.test_file, "r") as f:
    test_df = pd.read_csv(f)
    print("Test Dataframe Shape: " + str(test_df.shape))
    test_ds = prepare_dataset(test_df).shuffle(64)

  # Test specific model components.
  with tf.variable_scope("test_predictions_and_metrics"):
    _, acc_op, auc_op, update_acc_op, update_auc_op = predictions_and_metrics(
        negative_logits, positive_logits)

  saver = tf.train.Saver()

  test_itr = test_ds.make_one_shot_iterator()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(tf.initializers.local_variables())

    checkpoint = tf.train.latest_checkpoint(model_dir + "/save")
    saver.restore(sess, checkpoint)
    test_itr_handle = sess.run(test_itr.string_handle())
    while True:
      try:
        _, _ = sess.run([update_acc_op, update_auc_op],
                        feed_dict={handle: test_itr_handle})
      except tf.errors.OutOfRangeError:
        break
    test_acc, test_auc = sess.run([acc_op, auc_op])
    print("TEST ACCURACY: " + str(test_acc))
    print("TEST AUC: " + str(test_auc))
else:
  print("In TRAINING mode.")

  with tf.gfile.Open(FLAGS.train_file, "r") as f:
    train_df = pd.read_csv(f)
    print("Train Dataframe Shape: " + str(train_df.shape))
    train_dataset = prepare_dataset(train_df).shuffle(128).repeat()

  with tf.gfile.Open(FLAGS.validation_file, "r") as f:
    validation_df = pd.read_csv(f)
    print("Validation Dataframe Shape: " + str(validation_df.shape))
    validation_dataset = prepare_dataset(validation_df).shuffle(64)

  # Training specific model components.
  with tf.variable_scope("training_operations"):
    train_op, loss_op = train_operation(negative_logits, positive_logits)
  with tf.variable_scope("train_predictions_and_metrics"):
    _, train_acc_op, train_auc_op, train_update_acc_op, train_update_auc_op = predictions_and_metrics(
        negative_logits, positive_logits)
  with tf.variable_scope("validation_predictions_and_metrics"):
    _, val_acc_op, val_auc_op, val_update_acc_op, val_update_auc_op = predictions_and_metrics(
        negative_logits, positive_logits)

  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(tf.initializers.local_variables())

    train_writer = tf.summary.FileWriter(model_dir + "/train", sess.graph)
    validation_writer = tf.summary.FileWriter(model_dir + "/validation",
                                              sess.graph)

    training_iterator = train_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    best_auc = 0
    for batch_num in range(500):
      print("Batch: " + str(batch_num))

      batch_size = 32
      for i in range(batch_size):
        _, loss, train_acc, train_auc = sess.run(
            [train_op, loss_op, train_update_acc_op, train_update_auc_op],
            feed_dict={handle: training_handle})

        training_summary = tf.Summary(value=[
            tf.Summary.Value(tag="loss", simple_value=loss),
            tf.Summary.Value(tag="accuracy", simple_value=train_acc),
            tf.Summary.Value(tag="auc", simple_value=train_auc),
        ])
        train_writer.add_summary(training_summary, batch_num * batch_size + i)
        train_writer.flush()

      recent_aucs = collections.deque([], 3)

      sess.run(validation_iterator.initializer)
      for _ in range(32):
        _, _ = sess.run([val_update_acc_op, val_update_auc_op],
                        feed_dict={handle: validation_handle})
      val_acc, val_auc = sess.run([val_acc_op, val_auc_op])

      # Save best version
      if val_auc > best_auc:
        best_auc = val_auc
        saved_path = saver.save(
            sess, save_path, global_step=(batch_num + 1) * batch_size)

      # Early stopping
      if len(recent_aucs) >= 3 and all(
          val_auc < prev_auc for prev_auc in recent_aucs):
        break
      recent_aucs.append(val_auc)

      validation_summary = tf.Summary(value=[
          tf.Summary.Value(tag="accuracy", simple_value=val_acc),
          tf.Summary.Value(tag="auc", simple_value=val_auc),
      ])
      validation_writer.add_summary(validation_summary.SerializeToString(),
                                    (batch_num + 1) * batch_size)
      validation_writer.flush()
