import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import sys
import datetime


def distance(embeddings, prototype):
  return tf.map_fn(tf.norm, embeddings - prototype)


def neg_distance(embs, proto):
  return -distance(embs, proto)


def calculate_logits(embeddings, positive_prototype, negative_prototype):
  negative_logits = neg_distance(embeddings, negative_prototype)
  positive_logits = neg_distance(embeddings, positive_prototype)
  return tf.stack([negative_logits, positive_logits], axis=1)


def loss_from_embeddings(embeddings, positive_prototype, negative_prototype,
                         true_class):
  logits = calculate_logits(embeddings, positive_prototype, negative_prototype)
  return tf.losses.softmax_cross_entropy(
      tf.broadcast_to(tf.one_hot(true_class, 2), tf.shape(logits)), logits)


def prepare_dataset(data):
  data["text"] = data.text.fillna("")
  domains = data.domain.unique()
  print("DOMAINS: " + str(len(domains)))

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


with tf.gfile.Open(
    "gs://conversationai-models/resources/transfer_learning_data/many_communities_40_per_8_shot/train_cleaned_text.csv",
    "r") as f:
  train_df = pd.read_csv(f)
  print(train_df.shape)
  train_dataset = prepare_dataset(train_df).shuffle(128).repeat()

with tf.gfile.Open(
    "gs://conversationai-models/resources/transfer_learning_data/many_communities_40_per_8_shot/validation_cleaned_text.csv",
    "r") as f:
  validation_df = pd.read_csv(f)
  print("VALIDATION")
  print(validation_df.shape)
  validation_dataset = prepare_dataset(validation_df).shuffle(64)

embed = hub.Module(
    "https://tfhub.dev/google/universal-sentence-encoder-large/3")

dense_1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
dense_2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)

get_embeddings = lambda texts: dense_2(dense_1(embed(texts)))
get_prototype = lambda texts: tf.reduce_mean(get_embeddings(texts), 0)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle,
                                               train_dataset.output_types,
                                               train_dataset.output_shapes)
episode_batch = iterator.get_next()

negative_prototype = get_prototype(episode_batch["negative_supports"])
positive_prototype = get_prototype(episode_batch["positive_supports"])
negative_embeddings = get_embeddings(episode_batch["negative_queries"])
positive_embeddings = get_embeddings(episode_batch["positive_queries"])

negative_loss = loss_from_embeddings(negative_embeddings, positive_prototype,
                                     negative_prototype, 0)
positive_loss = loss_from_embeddings(positive_embeddings, positive_prototype,
                                     negative_prototype, 1)
loss = negative_loss + positive_loss
tf.summary.scalar("loss", loss)

optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss)

negative_logits = calculate_logits(negative_embeddings, positive_prototype,
                                   negative_prototype)
positive_logits = calculate_logits(positive_embeddings, positive_prototype,
                                   negative_prototype)
predict = lambda logits: tf.argmax(logits, axis=1)

negative_predictions = predict(negative_logits)
negative_labels = tf.fill(tf.shape(negative_predictions), 0)
positive_predictions = predict(positive_logits)
positive_labels = tf.fill(tf.shape(positive_predictions), 1)

labels = tf.concat([negative_labels, positive_labels], -1)
predictions = tf.concat([negative_predictions, positive_predictions], -1)
_, accuracy = tf.metrics.accuracy(labels, predictions)
_, roc_auc = tf.metrics.auc(labels, predictions)
#_, negative_accuracy = tf.metrics.accuracy(negative_labels,
#                                           negative_predictions)
#_, positive_accuracy = tf.metrics.accuracy(positive_labels,
#                                           positive_predictions)
#accuracy = tf.mean([negative_accuracy, positive_accuracy])
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("roc_auc", roc_auc)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  sess.run(tf.initializers.local_variables())

  merged = tf.summary.merge_all()

  st = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

  events_dir = "gs://conversationai-models/jjtan/transfer_learning/model/" + st
  print(events_dir)
  train_writer = tf.summary.FileWriter(events_dir + "/train", sess.graph)

  training_iterator = train_dataset.make_one_shot_iterator()
  training_handle = sess.run(training_iterator.string_handle())

  validation_writer = tf.summary.FileWriter(events_dir + "/validation",
                                            sess.graph)
  validation_iterator = validation_dataset.make_initializable_iterator()
  validation_handle = sess.run(validation_iterator.string_handle())

  for batch_num in range(5000):
    print("BATCH NUMBER: " + str(batch_num))

    batch_size = 32
    for i in range(batch_size):
      _, summary = sess.run([train, merged],
                            feed_dict={handle: training_handle})
      train_writer.add_summary(summary, batch_num * batch_size + i)
      train_writer.flush()

    sess.run(validation_iterator.initializer)
    agg_acc = []
    agg_auc = []
    for _ in range(32):
      acc, auc = sess.run([accuracy, roc_auc],
                          feed_dict={handle: validation_handle})
      agg_acc.append(acc)
      agg_auc.append(auc)
    validation_acc = np.mean(agg_acc)
    validation_auc = np.mean(agg_auc)
    validation_summary = tf.Summary(value=[
        tf.Summary.Value(tag="accuracy_1", simple_value=validation_acc),
        tf.Summary.Value(tag="roc_auc", simple_value=validation_auc),
    ])
    validation_writer.add_summary(validation_summary.SerializeToString(),
                                  (batch_num + 1) * batch_size)
    validation_writer.flush()
