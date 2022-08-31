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
from typing import List

import numpy as np
import tensorflow as tf
from keras.layers import TFOpLambda

import model_compression_toolkit as mct
import model_compression_toolkit.gptq.common.gptq_config
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.tpc_models.default_tpc.v1.tpc_keras import generate_keras_tpc
from model_compression_toolkit.gptq.keras.gptq_loss import multiple_tensors_mse_loss
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


def build_model(in_input_shape: List[int]) -> keras.Model:
    """
    This function generate a simple network to test GPTQ
    Args:
        in_input_shape: Input shape list

    Returns:

    """
    inputs = layers.Input(shape=in_input_shape)
    x = layers.Conv2D(32, 4, bias_initializer='glorot_uniform')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Conv2D(32, 8, bias_initializer='glorot_uniform')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.ReLU()(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


class ActivationGradientPTQBaseTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, is_gumbel=True, activation_quantization_method=QuantizationMethod.SYMMETRIC):
        super().__init__(unit_test,
                         input_shape=(1, 16, 16, 3))
        self.is_gumbel = is_gumbel
        self.activation_quantization_method = activation_quantization_method

    def get_tpc(self):
        tp_model = generate_test_tp_model({'weights_n_bits': 8,
                                           'activation_n_bits': 8,
                                           'activation_quantization_method': self.activation_quantization_method})
        return generate_keras_tpc(name="activation_gptq_test", tp_model=tp_model)

    def get_quantization_config(self):
        return mct.QuantizationConfig()

    def get_gptq_config(self):
        return model_compression_toolkit.gptq.common.gptq_config.GradientPTQConfig(5,
                                                                                   optimizer=tf.keras.optimizers.Adam(
                                                                                       learning_rate=0.0001),
                                                                                   optimizer_rest=tf.keras.optimizers.Adam(
                                                                                       learning_rate=0.0001),
                                                                                   sam_optimization=False,
                                                                                   loss=multiple_tensors_mse_loss,
                                                                                   rounding_type=model_compression_toolkit.gptq.common.gptq_config.RoundingType.GumbelRounding if self.is_gumbel else model_compression_toolkit.gptq.common.gptq_config.RoundingType.STE,
                                                                                   train_bias=True,
                                                                                   activation_parameters_learning=True,
                                                                                   optimizer_activation_params=tf.keras.optimizers.Adam(
                                                                                       learning_rate=0.0001))

    def create_networks(self):
        in_shape = self.get_input_shapes()[0][1:]
        return build_model(in_shape)

    def compare(self, ptq_model, model_float, input_x=None, quantization_info: UserInformation = None):
        raise NotImplementedError(f'{self.__class__} did not implement compare')

    def run_test(self, experimental_facade=False, experimental_exporter=False):
        x = self.generate_inputs()

        def representative_data_gen():
            return x

        model_float = self.create_networks()

        qc = self.get_quantization_config()
        tpc = self.get_tpc()
        if experimental_facade:
            ptq_model, quantization_info = mct.keras_post_training_quantization_experimental(model_float,
                                                                                             self.representative_data_gen,
                                                                                             target_kpi=self.get_kpi(),
                                                                                             core_config=self.get_core_config(),
                                                                                             target_platform_capabilities=self.get_tpc(),
                                                                                             new_experimental_exporter=experimental_exporter
                                                                                             )
            gptq_model, quantization_info = mct.keras_gradient_post_training_quantization_experimental(model_float,
                                                                                                       self.representative_data_gen,
                                                                                                       gptq_config=self.get_gptq_config(),
                                                                                                       target_kpi=self.get_kpi(),
                                                                                                       core_config=self.get_core_config(),
                                                                                                       target_platform_capabilities=self.get_tpc(),
                                                                                                       new_experimental_exporter=experimental_exporter
                                                                                                       )
        else:

            ptq_model, quantization_info = mct.keras_post_training_quantization(model_float, representative_data_gen,
                                                                                n_iter=self.num_calibration_iter,
                                                                                quant_config=qc,
                                                                                fw_info=DEFAULT_KERAS_INFO,
                                                                                network_editor=self.get_network_editor(),
                                                                                target_platform_capabilities=tpc)
            gptq_model, quantization_info = mct.keras_post_training_quantization(model_float, representative_data_gen,
                                                                                 n_iter=self.num_calibration_iter,
                                                                                 quant_config=qc,
                                                                                 fw_info=DEFAULT_KERAS_INFO,
                                                                                 network_editor=self.get_network_editor(),
                                                                                 gptq_config=self.get_gptq_config(),
                                                                                 target_platform_capabilities=tpc)

        self.compare(ptq_model, gptq_model, input_x=x, quantization_info=quantization_info)

    def verify_threshold_training(self, quantized_model, gptq_model):
            thresholds_diff = []
            # Skipping input layer activation since its threshold is not being trained
            for q_layer, g_layer in zip(quantized_model.layers[2:], gptq_model.layers[2:]):
                if isinstance(q_layer, TFOpLambda) and len(list(q_layer.inbound_nodes[0].call_kwargs.keys())) > 0:
                    q_max_level = q_layer.inbound_nodes[0].call_kwargs.get('max')
                    q_nbits = q_layer.inbound_nodes[0].call_kwargs.get('num_bits')
                    q_threshold = q_max_level / (2 ** q_nbits - 1) + q_max_level

                    g_max_level = g_layer.inbound_nodes[0].call_kwargs.get('max')
                    g_nbits = g_layer.inbound_nodes[0].call_kwargs.get('num_bits')
                    g_threshold = g_max_level / (2 ** g_nbits - 1) + g_max_level

                    thresholds_diff.append(not np.isclose(q_threshold - g_threshold, 0, atol=1e-5))

            # Check that at least one activation threshold has been changed due to training
            self.unit_test.assertTrue(np.any(thresholds_diff),
                                      "None of the activation thresholds have been changed due to GPTQ training")


class ActivationGradientPTQTest(ActivationGradientPTQBaseTest):
    def compare(self, quantized_model, gptq_model, input_x=None, quantization_info=None):
        self.verify_threshold_training(quantized_model, gptq_model)


class ActivationGradientPTQWeightedLossTest(ActivationGradientPTQBaseTest):
    def get_gptq_config(self):
        return model_compression_toolkit.gptq.common.gptq_config.GradientPTQConfig(5,
                                                                                   optimizer=tf.keras.optimizers.Adam(
                                                                                       learning_rate=0.0001),
                                                                                   optimizer_rest=tf.keras.optimizers.Adam(
                                                                                       learning_rate=0.0001),
                                                                                   loss=multiple_tensors_mse_loss,
                                                                                   train_bias=True,
                                                                                   use_jac_based_weights=True,
                                                                                   num_samples_for_loss=16,
                                                                                   norm_weights=False,
                                                                                   activation_parameters_learning=True,
                                                                                   optimizer_activation_params=tf.keras.optimizers.Adam(
                                                                                       learning_rate=0.0001))

    def compare(self, quantized_model, gptq_model, input_x=None, quantization_info=None):
        self.verify_threshold_training(quantized_model, gptq_model)


class ActivationGradientPTQLearnRateZeroTest(ActivationGradientPTQBaseTest):

    def get_gptq_config(self):
        return model_compression_toolkit.gptq.common.gptq_config.GradientPTQConfig(5,
                                                                                   optimizer=tf.keras.optimizers.SGD(
                                                                                       learning_rate=0.0),
                                                                                   optimizer_rest=tf.keras.optimizers.SGD(
                                                                                       learning_rate=0.0),
                                                                                   train_bias=True,
                                                                                   sam_optimization=False,
                                                                                   loss=multiple_tensors_mse_loss,
                                                                                   activation_parameters_learning=True,
                                                                                   optimizer_activation_params=tf.keras.optimizers.SGD(
                                                                                       learning_rate=0.0))

    def compare(self, quantized_model, gptq_model, input_x=None, quantization_info=None):
        thresholds_diff = []
        # Skipping input layer activation since its threshold is not being trained
        for q_layer, g_layer in zip(quantized_model.layers[2:], gptq_model.layers[2:]):
            if isinstance(q_layer, TFOpLambda) and len(list(q_layer.inbound_nodes[0].call_kwargs.keys())) > 0:
                q_max_level = q_layer.inbound_nodes[0].call_kwargs.get('max')
                q_nbits = q_layer.inbound_nodes[0].call_kwargs.get('num_bits')
                q_threshold = q_max_level / (2 ** q_nbits - 1) + q_max_level

                g_max_level = g_layer.inbound_nodes[0].call_kwargs.get('max')
                g_nbits = g_layer.inbound_nodes[0].call_kwargs.get('num_bits')
                g_threshold = g_max_level / (2 ** g_nbits - 1) + g_max_level

                thresholds_diff.append(not np.isclose(q_threshold - g_threshold, 0, atol=1e-5))

        # Check that activation thresholds didn't change
        self.unit_test.assertFalse(np.any(thresholds_diff),
                                   "None of the activation thresholds shouldn't have change from the quantized_model "
                                   "thresholds, since training lr is 0")
