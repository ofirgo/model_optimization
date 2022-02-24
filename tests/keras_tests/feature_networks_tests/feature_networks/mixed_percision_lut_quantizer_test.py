# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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


import numpy as np
import tensorflow as tf
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

import model_compression_toolkit as mct
from model_compression_toolkit.common.mixed_precision.kpi import KPI
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.user_info import UserInformation
from tests.common_tests.base_feature_test import BaseFeatureNetworkTest
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class MixedPercisionWithLUTQuantizer2BitTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        qc = mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE, mct.QuantizationErrorMethod.MSE,
                                    mct.QuantizationMethod.POWER_OF_TWO, mct.QuantizationMethod.POWER_OF_TWO,
                                    relu_unbound_correction=True, weights_bias_correction=True,
                                    weights_per_channel_threshold=True, input_scaling=True,
                                    activation_channel_equalization=True)

        return MixedPrecisionQuantizationConfig(qc, weights_n_bits=[2, 8, 4], num_of_images=1, lut_nbits=[2])

    def get_bit_widths_config(self):
        return None

    def get_kpi(self):
        # kpi is for 2 bits on average
        return KPI(2544200 * 2 / 8)

    def get_input_shapes(self):
        return [[self.val_batch_size, 224, 244, 3]]

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(30, 40)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(50, 40)(x)
        outputs = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # This is a basic test just to verify that the is running without crashing (not actually checking that the LUT quantizer ran)
        self.unit_test.assertTrue(True)
