# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from abc import abstractmethod
from typing import Union, Dict, List

import tensorflow as tf

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.core.keras.constants import KERNEL
from model_compression_toolkit.gptq.common.gptq_constants import WEIGHTS_QUANTIZATION_PARAMS, PTQ_THRESHOLD, AUXVAR

from model_compression_toolkit.quantizers_infrastructure import TrainableQuantizerWeightsConfig, \
    TrainableQuantizerActivationConfig, BaseKerasTrainableQuantizer

if FOUND_TF:

    class BaseKerasGPTQTrainableQuantizer(BaseKerasTrainableQuantizer):
        """
        A base class for trainable Keras quantizer for GPTQ.
        """

        def __init__(self,
                     quantization_config: Union[TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig]):
            """
            Initializes BaseKerasGPTQTrainableQuantizer object.

            Args:
                quantization_config: quantizer config class contains all the information about a quantizer configuration.
            """

            super().__init__(quantization_config)

            self.quantizer_parameters = None

        def update_layer_quantization_params(self, layer) -> (Dict[str, tf.Tensor], Dict[str, Dict], Dict):
            """
            A Function to calculate the needed change in attributes in NodeQuantizationConfig after retraining.

            Returns:
                3 dictionaries describing the change in layer's weights, weights config, activation config
                that changed during GPTQ retraining.
                Keys must match NodeQuantizationConfig attributes

            """
            weights = {}
            for weight, quantizer_vars, quantizer in layer.get_weights_vars():
                weights.update({weight: quantizer(training=False, inputs=quantizer_vars)})

            quant_config = {WEIGHTS_QUANTIZATION_PARAMS: self.get_quant_config()}

            return weights, quant_config, {}

        def get_aux_variable(self) -> List[tf.Tensor]:
            """
            This function return a list with the quantizer's quantization auxiliary variables.

            Returns: A list with the quantization auxiliary variables.

            """

            return []

        def get_quantization_variable(self) -> List[tf.Tensor]:
            """
            This function return a list with the quantizer's quantization parameters variables.

            Returns: A list with the quantization parameters.

            """

            return []

        @abstractmethod
        def get_quant_config(self):
            """
            Returns the config used to edit NodeQuantizationConfig after GPTQ retraining.

            Returns:
                A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
                Keys must match NodeQuantizationConfig attributes.

            """
            raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                                 f'quantizer\'s get_quant_config.')

else:
    class BaseKerasGPTQTrainableQuantizer(BaseKerasTrainableQuantizer):
        def __init__(self,
                     quantization_config: Union[TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig]):

            super().__init__(quantization_config)
            Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using BaseKerasGPTQTrainableQuantizer. '
                            'Could not find Tensorflow package.')  # pragma: no cover