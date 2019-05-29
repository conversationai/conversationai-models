import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import sys

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    "module_spec",
    "https://tfhub.dev/google/universal-sentence-encoder-large/3",
    "The url of the TF Hub sentence encoding module to use.")

TRAIN_DATA = {
    "positive_supports":
        np.array([["This is english.", "I like to ski"],
                  ["Second Episode", "Some more words that sounds good."]]),
    "negative_supports":
        np.array([["Whoasdfklj", "asdfkjvn"], ["asdf", "you wonk tea?"]]),
    "positive_queries":
        np.array(
            [["Too much fun.", "Apples are great!"],
             ["Chinese is not like English", "Dogs and cats are different."]]),
    "negative_queries":
        np.array([["Kan wo ba.", "Funtillkick"],
                  ["panionio", "asdfads asfjni fmek"]]),
}

TEST_FEATURES = {
    "positive_supports":
        np.array([["A random english phrase", "I like to ski"],
                  ["Second Episode", "Some more words that sounds good."]]),
    "negative_supports":
        np.array([["Whoasdfklj", "asdfkjvn"], ["asdf", "you wonk tea?"]]),
    "query":
        np.array([["Candles can blow out.", "Afakdj!"],
                  ["Carrots and onions go well together.", "as#@kjf afsdi"]]),
}
TEST_LABELS = {}

NUM_EPISODES = 10000
BATCH_SIZE = 4
N_SHOT = 8


def split_class_data(class_data):
  class_support = class_data.sample(N_SHOT)
  class_query = class_data[~class_data.isin(class_support).all(1)]
  return (class_support, class_query)


def split_domain_data(domain_data):
  positive_data = domain_data[domain_data["label"] == 1]
  positive_support, positive_query = split_class_data(positive_data)
  negative_data = domain_data[domain_data["label"] == 0]
  negative_support, negative_query = split_class_data(negative_data)
  results = [positive_support, positive_query, negative_support, negative_query]
  return tuple([result["text"].values for result in results])


def distance(embeddings, prototype):
  return tf.map_fn(tf.norm, embeddings - prototype)


def neg_distance(embs, proto):
  return -distance(embs, proto)


def loss_from_embeddings(embeddings, positive_prototype, negative_prototype,
                         true_class):
  negative_logits = neg_distance(embeddings, negative_prototype)
  positive_logits = neg_distance(embeddings, positive_prototype)
  logits = tf.stack([negative_logits, positive_logits], axis=1)
  return tf.losses.softmax_cross_entropy(
      tf.broadcast_to(tf.one_hot(true_class, 2), tf.shape(logits)), logits)


def main(argv):
  del argv  # unused

  with tf.gfile.Open(
      "gs://kaggle-model-experiments/resources/transfer_learning_data/many_communities.csv",
      "r") as f:
    data = pd.read_csv(f)

  domains = data.domain.unique()

  positive_supports = []
  positive_queries = []
  negative_supports = []
  negative_queries = []
  for domain in domains:
    domain_data = data[data["domain"] == domain]
    positive_support, positive_query, negative_support, negative_query = split_domain_data(
        domain_data)
    positive_supports.append(positive_support)
    positive_queries.append(positive_query)
    negative_supports.append(negative_support)
    negative_queries.append(negative_query)

  train_dataset = tf.data.Dataset.from_tensor_slices({
      "positive_supports": np.array(positive_supports),
      "negative_supports": np.array(negative_supports),
      "positive_queries": np.array(positive_queries),
      "negative_queries": np.array(negative_queries)
  })
  iterator = train_dataset.shuffle(128).repeat(10).make_one_shot_iterator()
  episode = iterator.get_next()

  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  dense_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
  dense_2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)

  get_embeddings = lambda texts: dense_2(dense_1(embed(texts)))

  negative_prototype = tf.reduce_mean(
      get_embeddings(episode["negative_supports"]), 0)
  positive_prototype = tf.reduce_mean(
      get_embeddings(episode["positive_supports"]), 0)
  negative_embeddings = get_embeddings(episode["negative_queries"])
  positive_embeddings = get_embeddings(episode["positive_queries"])

  negative_loss = loss_from_embeddings(negative_embeddings, positive_prototype,
                                       negative_prototype, 0)
  positive_loss = loss_from_embeddings(positive_embeddings, positive_prototype,
                                       negative_prototype, 1)
  loss = negative_loss + positive_loss
  tf.summary.scalar("loss", loss)

  optimizer = tf.train.AdamOptimizer(0.001)
  train = optimizer.minimize(loss)

  predict = lambda embs: tf.nn.softmax([
      neg_distance(embs, negative_prototype),
      neg_distance(embs, positive_prototype)
  ])
  prediction = predict(negative_embeddings)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("/tmp/train", sess.graph)

    for i in range(NUM_EPISODES):
      _, summary = sess.run([train, merged])
      train_writer.add_summary(summary, i)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
