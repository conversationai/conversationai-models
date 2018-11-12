# coding=utf-8
# Copyright 2018 The Conversation-AI.github.io Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
