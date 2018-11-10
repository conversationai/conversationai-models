py_library(
  name = "tf_trainer",
  srcs = [
    "model_trainer.py",
    "text_preprocessor.py",
  ],
  deps = [
    ":data_input",
    ":base_model",
    ":types",
  ],
)

py_library(
  name = "types",
  srcs = [
    "types.py",
  ],
)

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
