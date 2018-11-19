py_library(
  name = "types",
  srcs = [
    "types.py",
  ],
)

py_library(
  name = "model_trainer",
  srcs = [
    "model_trainer.py",
  ],
  deps = [
    ":data_input",
    ":base_model",
    ":types",
    ":text_preprocessor",
  ],
)

py_library(
  name = "token_embedding_index",
  srcs = [
    "token_embedding_index.py",
  ],
  deps = [":types",":base_model"],
)

py_test(
  name = "token_embedding_index_test",
  srcs = ["token_embedding_index_test.py",],
  deps = [":token_embedding_index", ":types",],
  data = ["//testdata:cats_and_dogs_onehot.vocab.txt",],
)

py_library(
  name = "text_preprocessor",
  srcs = [
    "text_preprocessor.py",
  ],
  deps = [":types",":base_model"],
)

# TODO: add a test for the text preprocessor.
# py_test(
#   name = "text_preprocessor_test",
#   srcs = ["text_preprocessor_test.py",],
#   deps = [":text_preprocessor", ":types",],
#   data = ["testdata/cats_and_dogs_onehot.vocab.txt",],
# )

py_library(
  name = "base_model",
  srcs = [
    "base_model.py",
    "base_keras_model.py",
  ],
  deps = [":types",],
)

py_test(
  name = "base_keras_model_test",
  srcs = ["base_keras_model_test.py",],
  deps = [":base_model", ":types",],
)

py_library(
  name = "data_input",
  srcs = [
    "tfrecord_input.py",
    "tfrecord_simple.py",
    "dataset_input.py",
  ],
  deps = [":types",],
)

py_test(
  name = "tfrecord_input_test",
  srcs = ["tfrecord_input_test.py",],
  deps = [":data_input", ":types",],
)

py_test(
  name = "tfrecord_simple_test",
  srcs = ["tfrecord_simple_test.py",],
  deps = [":data_input", ":types",],
)

py_library(
  name = "cnn_spec_parser",
  srcs = ["cnn_spec_parser.py",],
  deps = [":types",],
)

py_test(
  name = "cnn_spec_parser_test",
  srcs = ["cnn_spec_parser_test.py",],
  deps = [":cnn_spec_parser", ":types",],
)
