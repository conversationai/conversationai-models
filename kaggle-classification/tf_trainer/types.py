"""Types for the tf_trainer module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import NewType, Union, Dict, Tuple

# Type aliases for convenience.

Tensor = Union[tf.Tensor, tf.SparseTensor]
TensorDict = Dict[str, Tensor]
TensorOrTensorDict = Union[tf.Tensor, TensorDict]
FeatureAndLabelTensors = Tuple[TensorOrTensorDict, TensorOrTensorDict]

# New Types.

Path = NewType("Path", str)
