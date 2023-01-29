# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
import copy
import unittest
from typing import List

import numpy as np
from model_compression_toolkit import get_keras_gptq_config, keras_post_training_quantization, keras_gradient_post_training_quantization_experimental, \
    QuantizationConfig, QuantizationErrorMethod, GradientPTQConfig, RoundingType, CoreConfig
import tensorflow as tf
from model_compression_toolkit.gptq.keras.gptq_loss import multiple_tensors_mse_loss
import model_compression_toolkit as mct

layers = tf.keras.layers
SHAPE = [1, 16, 16, 3]


def build_model(in_input_shape: List[int]) -> tf.keras.Model:
    """
    This function generate a simple network to test GPTQ
    Args:
        in_input_shape: Input shape list

    Returns:

    """
    inputs = layers.Input(shape=in_input_shape)
    x = layers.Conv2D(3, 4)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Conv2D(7, 8)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.ReLU()(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def random_datagen():
    return [np.random.random(SHAPE)]


def random_datagen_experimental():
    yield [np.random.random(SHAPE)]


class TestGetGPTQConfig(unittest.TestCase):

    def test_get_keras_gptq_config(self):
        qc = QuantizationConfig(QuantizationErrorMethod.MSE,
                                QuantizationErrorMethod.MSE,
                                weights_bias_correction=False)  # disable bias correction when working with GPTQ
        cc = CoreConfig(quantization_config=qc)
        gptq_configurations = [GradientPTQConfig(1, optimizer=tf.keras.optimizers.RMSprop(),
                                                 optimizer_rest=tf.keras.optimizers.RMSprop(), train_bias=True,
                                                 loss=multiple_tensors_mse_loss),
                               GradientPTQConfig(1, optimizer=tf.keras.optimizers.Adam(),
                                                 optimizer_rest=tf.keras.optimizers.Adam(), train_bias=True,
                                                 loss=multiple_tensors_mse_loss),
                               ]

        gptqv2_configurations = [get_keras_gptq_config(n_epochs=1, optimizer=tf.keras.optimizers.Adam())]



        for i, gptq_config in enumerate(gptq_configurations):
            keras_post_training_quantization(in_model=build_model(SHAPE[1:]),
                                             representative_data_gen=random_datagen,
                                             n_iter=1,
                                             quant_config=qc,
                                             gptq_config=gptq_config)

        for i, gptq_config in enumerate(gptqv2_configurations):
            keras_gradient_post_training_quantization_experimental(in_model=build_model(SHAPE[1:]),
                                                                   representative_data_gen=random_datagen_experimental,
                                                                   core_config=cc,
                                                                   gptq_config=gptq_config)

    def test_get_keras_unsupported_configs_raises(self):

        with self.assertRaises(Exception) as e:
            GradientPTQConfig(1,
                              optimizer=tf.keras.optimizers.Adam(),
                              optimizer_rest=tf.keras.optimizers.Adam(),
                              train_bias=True,
                              quantization_parameters_learning=True,
                              loss=multiple_tensors_mse_loss)
        self.assertEqual('Quantization parameters learning is not supported with STE rounding.', str(e.exception))


if __name__ == '__main__':
    unittest.main()
