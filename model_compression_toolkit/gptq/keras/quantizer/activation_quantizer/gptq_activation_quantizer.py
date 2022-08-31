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
import numpy as np

from model_compression_toolkit.core.keras.quantizer.base_quantizer import BaseTrainableQuantizer
from model_compression_toolkit.gptq.keras.quantizer import quant_utils as qutils
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow.python.framework.tensor_shape import TensorShape
from typing import Dict, Any, List
from model_compression_toolkit.core.common.constants import THRESHOLD, SIGNED
from model_compression_toolkit.gptq.common import gptq_constants


def symmetric_constrained_quantizer(input_tensor: tf.Tensor,
                                    activation_threshold: tf.Variable,
                                    num_bits: int,
                                    signed: bool) -> tf.Tensor:
    """
    Quantize a tensor symmetrically with a given threshold.

    Args:
        input_tensor: Tensor to quantize. values of this tensor are not changed during gptq.
        activation_threshold: The activation quantization threshold variable.
        num_bits: Num of bits to use.
        signed: Signedness of the quantization range.

    Returns:
        A quantized tensor.
    """

    delta = qutils.calculate_delta(tf.convert_to_tensor(activation_threshold), num_bits, signed)
    tensor_q = qutils.ste_round(input_tensor / delta)
    min_int = -int(signed) * (2 ** (num_bits - int(signed)))
    max_int = (2 ** (num_bits - int(signed))) - 1
    return delta * qutils.clip(tensor_q, max_val=max_int, min_val=min_int)


class GPTQActivationQuantizer(BaseTrainableQuantizer):
    """
    Trainable symmetric quantizer to quantize a layer activation.
    """
    ACTIVATION_PTQ_THRESHOLD = "_activation_ptq_threshold"

    def __init__(self,
                 num_bits: int,
                 signed: bool,
                 threshold_value: np.ndarray):
        """
        Initialize a GPTQActivationQuantizer object with trainable threshold to use
        for the quantization.

        Args:
            num_bits: Number of bits to use for the quantization.
            signed: Signedness to use for the quantization range.
            threshold_value: Threshold to use for the quantization.
        """

        self.threshold_value = threshold_value
        self.num_bits = num_bits
        self.signed = signed
        self.quantizer_parameters = {}

    def build(self,
              tensor_shape: TensorShape,
              name: str,
              layer: QuantizeWrapper) -> Dict[str, tf.Variable]:
        """
        Add threshold variable to layer.

        Args:
            tensor_shape: Tensor shape the quantizer quantize.
            name: Prefix of variables names.
            layer: Layer to add the variables to. The variables are saved
            in the layer's scope.

        Returns:
            Dictionary of new variables.
        """

        activation_threshold = layer.add_weight(
            name + gptq_constants.ACTIVATION_THRESHOLD,
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True)
        activation_threshold.assign(self.threshold_value)

        self.quantizer_parameters = {gptq_constants.ACTIVATION_THRESHOLD: activation_threshold}
        return self.quantizer_parameters

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool,
                 weights: Dict[str, tf.Variable],
                 **kwargs: Dict[str, Any]):
        """
        Quantize a tensor.

        Args:
            inputs: Input tensor to quantize.
            training: Whether the graph is in training mode.
            weights: Dictionary of parameters the quantizer can use to quantize the tensor.
            **kwargs: Additional variables the quantizer may receive.

        Returns:
            The quantized tensor.
        """
        activation_threshold = weights[gptq_constants.ACTIVATION_THRESHOLD]

        return symmetric_constrained_quantizer(inputs,
                                               activation_threshold,
                                               self.num_bits,
                                               self.signed)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns: Configuration of TrainableQuantizer.
        """

        return {
            'num_bits': self.num_bits,
            'signed': self.signed,
            'threshold_value': self.threshold_value
        }

    def get_quant_config(self, layer) -> Dict[str, np.ndarray]:
        """
        Returns the config used to edit NodeQuantizationConfig after GPTQ retraining

        Args:
            layer: quantized layer

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes

        """
        threshold = self.quantizer_parameters[gptq_constants.ACTIVATION_THRESHOLD]

        return {THRESHOLD: threshold.numpy(),
                SIGNED: self.signed}

    def get_trainable_parameters(self):
        """
        A function to get a list trainable of trainable parameters of the quantizer for GPTQ retraining

        Returns:
            A list of trainable Tensors

        """
        return [t for t in self.quantizer_parameters.values() if t.trainable]

    def get_quantization_variable(self) -> List[tf.Tensor]:
        """
         This function return a list of quantizer parameters.
         Returns: A list of the quantizer parameters

         """
        return [self.quantizer_parameters[gptq_constants.ACTIVATION_THRESHOLD]]

    def __eq__(self, other: Any) -> bool:
        """
        Check if equals to another object.
        Args:
            other: Other object to compare.

        Returns:
            Whether they are equal or not.
        """
        if not isinstance(other, GPTQActivationQuantizer):
            return False

        return (self.num_bits == other.num_bits and
                self.signed == other.signed and
                self.threshold_value == other.threshold_value)

    def __ne__(self, other: Any) -> bool:
        """
        Check if not equals to another object.
        Args:
            other: Other object to compare.

        Returns:
            Whether they are differ or not.
        """
        return not self.__eq__(other)