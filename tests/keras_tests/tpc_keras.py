# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from keras.engine.input_layer import InputLayer
import model_compression_toolkit as mct

from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model, \
    generate_mixed_precision_test_tp_model, generate_tp_model_with_activation_mp, generate_test_tpc
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_keras_tpc
from keras.layers import Conv2DTranspose

tp = mct.target_platform


def get_tpc(name, weight_bits=8, activation_bits=8,
            weights_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO):
    tp_model = generate_test_tp_model({'weights_n_bits': weight_bits,
                                       'activation_n_bits': activation_bits,
                                       'weights_quantization_method': weights_quantization_method})
    return generate_keras_tpc(name=name, tp_model=tp_model)


def get_16bit_tpc(name):
    tp_model = generate_test_tp_model({'weights_n_bits': 16,
                                       'activation_n_bits': 16})
    return generate_keras_tpc(name=name, tp_model=tp_model)


def get_16bit_tpc_per_tensor(name):
    tp_model = generate_test_tp_model({'weights_n_bits': 16,
                                       'activation_n_bits': 16,
                                       "weights_per_channel_threshold": False})
    return generate_keras_tpc(name=name, tp_model=tp_model)


def get_quantization_disabled_keras_tpc(name):
    tp = generate_test_tp_model({'enable_weights_quantization': False,
                                 'enable_activation_quantization': False})
    return generate_keras_tpc(name=name, tp_model=tp)


def get_activation_quantization_disabled_keras_tpc(name):
    tp = generate_test_tp_model({'enable_activation_quantization': False})
    return generate_keras_tpc(name=name, tp_model=tp)


def get_weights_quantization_disabled_keras_tpc(name):
    tp = generate_test_tp_model({'enable_weights_quantization': False})
    return generate_keras_tpc(name=name, tp_model=tp)


def get_weights_only_mp_tpc_keras(base_config, mp_bitwidth_candidates_list, name):
    mp_tp_model = generate_mixed_precision_test_tp_model(base_cfg=base_config,
                                                         mp_bitwidth_candidates_list=mp_bitwidth_candidates_list)
    return generate_keras_tpc(name=name, tp_model=mp_tp_model)


def get_tpc_with_activation_mp_keras(base_config, mp_bitwidth_candidates_list, name):
    mp_tp_model = generate_tp_model_with_activation_mp(base_cfg=base_config,
                                                       mp_bitwidth_candidates_list=mp_bitwidth_candidates_list)

    op_sets_to_layer_add = {
        "Input": [InputLayer],
        "Conv": [Conv2DTranspose, tf.nn.conv2d_transpose]
    }

    return generate_test_tpc(name=name,
                             tp_model=mp_tp_model,
                             base_tpc=generate_keras_tpc(name=f"base_{name}", tp_model=mp_tp_model),
                             op_sets_to_layer_add=op_sets_to_layer_add)
