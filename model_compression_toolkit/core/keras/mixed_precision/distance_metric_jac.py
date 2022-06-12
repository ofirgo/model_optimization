# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
# ==============================================================================
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.layers.base import Layer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit.core.keras.gradient_ptq.graph_info import get_trainable_parameters
from model_compression_toolkit.core.keras.quantizer.mixed_precision.selective_quantize_config import \
    SelectiveQuantizeConfig


# def distance_metric_jac(distance_metrix: Tensor,
#                         layer_output_tensor: Tensor):
#     with tf.GradientTape() as g:
#         x = layer_output_tensor
#         g.watch(x)
#         y = distance_metrix
#     jacobian = g.jacobian(y, x)
#     return jacobian

def distance_metric_jac(model,
                        images,
                        interest_points):

    with tf.GradientTape(persistent=True) as g:
        # g.watch(model.outputs[0])  # TODO: not working since this is not a Tensor but a KerasTensor, but we need to somehow track the outputs of each interest point
        out = model(images)

    model_output = out[-1]
    jacobians = []
    for i in range(len(interest_points)):
        jacobian = g.jacobian(model_output, model.outputs[i])
        jacobians.append(jacobian)
    return jacobians