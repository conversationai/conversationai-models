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
        ":base_model",
        ":data_input",
        ":text_preprocessor",
        ":types",
    ],
)

py_library(
    name = "token_embedding_index",
    srcs = [
        "token_embedding_index.py",
    ],
    deps = [
        ":base_model",
        ":types",
    ],
)

py_test(
    name = "token_embedding_index_test",
    srcs = ["token_embedding_index_test.py"],
    data = ["//testdata:cats_and_dogs_onehot.vocab.txt"],
    deps = [
        ":token_embedding_index",
        ":types",
    ],
)

py_library(
    name = "text_preprocessor",
    srcs = [
        "text_preprocessor.py",
    ],
    deps = [
        ":base_model",
        ":token_embedding_index",
        ":types",
    ],
)

py_test(
    name = "text_preprocessor_test",
    srcs = ["text_preprocessor_test.py"],
    data = [
        "//testdata:cats_and_dogs_onehot.vocab.txt",
    ],
    deps = [
        ":text_preprocessor",
        ":types",
    ],
)

py_library(
    name = "base_model",
    srcs = [
        "base_model.py",
    ],
    deps = [":types"],
)

py_library(
    name = "data_input",
    srcs = [
        "dataset_input.py",
        "tfrecord_input.py",
        ":base_model",
    ],
    deps = [":types"],
)

py_test(
    name = "tfrecord_input_test",
    srcs = ["tfrecord_input_test.py"],
    deps = [
        ":data_input",
        ":types",
    ],
)

py_library(
    name = "cnn_spec_parser",
    srcs = ["cnn_spec_parser.py"],
    deps = [":types"],
)

py_test(
    name = "cnn_spec_parser_test",
    srcs = ["cnn_spec_parser_test.py"],
    deps = [
        ":cnn_spec_parser",
        ":types",
    ],
)

py_library(
    name = "episodic_tfrecord_input",
    srcs = ["episodic_tfrecord_input.py"],
    deps = [
        ":types",
        ":base_model",
        ":data_input",
    ],
)
