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
import tensorflow as tf
import numpy as np

from model_compression_toolkit.gptq.keras.quantizer import quant_utils as qutils
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow.python.framework.tensor_shape import TensorShape
from typing import Dict, Any, List
from model_compression_toolkit.core.common.constants import THRESHOLD, GUMBEL_MAX_ITER
from model_compression_toolkit.gptq.common import gptq_constants
from model_compression_toolkit.core.keras.quantizer.base_quantizer import BaseTrainableQuantizer


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 20, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t: int):
        """
        Cosine annealing scheduler for temperature b.

        Args:
            t: The current time step.

        Returns: Scheduled temperature.

        """

        ind = tf.cast(t < self.start_decay, tf.float32)

        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)

        return self.start_b * ind + (1 - ind) * (
                self.end_b + (self.start_b - self.end_b) * tf.math.maximum(0.0, (1 - rel_t)))


class SymmetricSoftRounding(BaseTrainableQuantizer):
    """
    Trainable symmetric quantizer to optimize the rounding of the quantized values using soft quantization method.
    """

    PTQ_THRESHOLD = "_ptq_threshold"
    SCALE_PTQ = "_scale"

    def __init__(self, num_bits: int,
                 per_axis: bool,
                 signed: bool,
                 quantization_parameter_learning: bool,
                 threshold_values: np.ndarray,
                 quantization_axis: int = -1,
                 max_iteration: int = GUMBEL_MAX_ITER):
        """
        Initialize a SymmetricSoftRounding object with parameters to use
        for the quantization.

        Args:
            num_bits: Number of bits to use for the quantization.
            per_axis: Whether to quantize per-channel or per-tensor.
            signed: Signedness to use for the quantization range.
            threshold_values: Threshold to use for the quantization.
            quantization_axis: Axis of tensor to use for the quantization.
        """

        super().__init__()
        self.max_iteration = max_iteration
        self.num_bits = num_bits
        self.per_axis = per_axis
        self.signed = signed
        self.quantization_parameter_learning = quantization_parameter_learning
        self.quantization_axis = quantization_axis
        self.threshold_shape = np.asarray(threshold_values).shape
        self.threshold_values = np.reshape(np.asarray(threshold_values), [-1]) if self.per_axis else float(
            threshold_values)
        self.k_threshold = len(self.threshold_values) if self.per_axis else 1
        self.gamma = -0.1
        self.zeta = 1.1
        self.beta = 2 / 3
        self.quantizer_parameters = {}
        self.linear_decay = LinearTempDecay((self.max_iteration * 1024) // 32)

    def build(self,
              tensor_shape: TensorShape,
              name: str,
              layer: QuantizeWrapper) -> Dict[str, tf.Variable]:
        """
        Add variables to the quantizer.

        Args:
            tensor_shape: Tensor shape the quantizer quantize.
            name: Prefix of variables names.
            layer: Layer to add the variables to. The variables are saved
            in the layer's scope.

        Returns:
            Dictionary of new variables.
        """

        super().build(tensor_shape, name, layer)

        if self.per_axis:
            input_shape = tensor_shape
            n_axis = len(input_shape)
            quantization_axis = n_axis + self.quantization_axis if self.quantization_axis < 0 else \
                self.quantization_axis
            reshape_shape = [self.k_threshold if i == quantization_axis else 1 for i in range(n_axis)]
        else:
            reshape_shape = [self.k_threshold]

        ar_iter = layer.add_weight(
            name + gptq_constants.GPTQ_ITER,
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False)

        ptq_threshold_tensor = layer.add_weight(
            name + self.PTQ_THRESHOLD,
            shape=reshape_shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        ptq_threshold_tensor.assign(self.threshold_values.reshape(reshape_shape))

        w = getattr(layer.layer, name)
        auxvar_tensor = layer.add_weight(
            name + gptq_constants.AUXVAR,
            shape=[*w.shape],
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True)

        delta = qutils.calculate_delta(ptq_threshold_tensor, self.num_bits, self.signed)
        w_floor = tf.floor(w / delta)
        rest = (w / delta) - w_floor  # rest of rounding [0, 1)
        alpha = -qutils.safe_log((self.zeta - self.gamma) / (rest - self.gamma) - 1, 1e-16)  # => sigmoid(alpha) = rest

        auxvar_tensor.assign(alpha)

        self.quantizer_parameters.update({gptq_constants.AUXVAR: auxvar_tensor,
                                          self.PTQ_THRESHOLD: ptq_threshold_tensor,
                                          gptq_constants.GPTQ_ITER: ar_iter})

        if self.quantization_parameter_learning:
            scale = layer.add_weight(
                name + self.SCALE_PTQ,
                shape=self.k_threshold,
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True)
            self.quantizer_parameters.update({self.SCALE_PTQ: scale})

        return self.quantizer_parameters

    def get_quantization_variable(self) -> List[tf.Tensor]:
        """
        Returns: A list of the quantization parameters (if there are defined parameters for the quantizer).
        """

        if self.quantization_parameter_learning:
            return [self.quantizer_parameters[self.SCALE_PTQ]]
        else:
            return []

    def get_regularization(self):
        st = self.get_soft_targets()
        b = self.linear_decay(self.ar_iter.value())
        return tf.reduce_sum(1 - tf.pow(tf.math.abs(st - .5) * 2, b))

    def get_trainable_parameters(self) -> List[tf.Tensor]:
        """
        A function to get a list trainable of trainable parameters of the quantizer for GPTQ retraining

        Returns:
            A list of trainable Tensors

        """
        return [t for t in self.quantizer_parameters.values() if t.trainable]

    def get_config(self) -> Dict[str, Any]:
        """
        Returns: Configuration of SymmetricSoftRounding.
        """

        return {
            'num_bits': self.num_bits,
            'per_axis': self.per_axis,
        }

    def get_soft_targets(self):
        return qutils.clip(
            tf.sigmoid(self.quantizer_parameters[gptq_constants.AUXVAR]) * (self.zeta - self.gamma) + self.gamma, 1, 0)

    def get_aux_variable(self):
        return self.quantizer_parameters[gptq_constants.AUXVAR]

    def __call__(self, inputs: tf.Tensor,
                 training: bool,
                 weights: Dict[str, tf.Variable],
                 **kwargs: Dict[str, Any]):
        """
        Quantize a tensor.

        Args:
            inputs: Input tensor to quantize.
            training: Whether the graph is in training mode.
            weights: Dictionary of weights the quantizer can use to quantize the tensor.
            **kwargs: Additional variables the quantizer may receive.

        Returns:
            The quantized tensor.
        """

        self.ar_iter = weights[gptq_constants.GPTQ_ITER]
        ptq_threshold_tensor = weights[self.PTQ_THRESHOLD]

        if self.per_axis:
            input_shape = inputs.shape
            n_axis = len(input_shape)
            quantization_axis = n_axis + self.quantization_axis if self.quantization_axis < 0 else \
                self.quantization_axis
            reshape_shape = [-1 if i == quantization_axis else 1 for i in range(n_axis)]

            ##########################################################
            # Calculate soft rounding targets and optimized threshold
            ##########################################################
            ptq_threshold_tensor_hat = tf.reshape(ptq_threshold_tensor, reshape_shape)
            aux_var = self.get_soft_targets()

            #####################################################
            # Gumbel Softmax
            #####################################################
            if training:
                self.ar_iter.assign_add(1.0)
            else:
                aux_var = tf.cast(weights[gptq_constants.AUXVAR] >= 0, tf.float32)

            #####################################################
            # Quantized Input
            #####################################################
            q_tensor = qutils.rounding_symmetric_quantizer(input_tensor=inputs,
                                                           auxvar_tensor=aux_var,
                                                           threshold_tensor=ptq_threshold_tensor_hat,
                                                           num_bits=self.num_bits,
                                                           signed=self.signed,
                                                           power_of_two=False)

            if self.quantization_parameter_learning:
                scale = tf.reshape(self.quantizer_parameters[self.SCALE_PTQ], reshape_shape)
                q_tensor *= scale

            return q_tensor
        else:
            return qutils.rounding_symmetric_quantizer(input_tensor=inputs,
                                                       auxvar_tensor=weights[gptq_constants.AUXVAR],
                                                       threshold_tensor=ptq_threshold_tensor.value(),
                                                       num_bits=self.num_bits,
                                                       signed=self.signed,
                                                       power_of_two=False)

    def get_quant_config(self, layer) -> Dict[str, np.ndarray]:
        """
        Returns the config used to edit NodeQuantizationConfig after GPTQ retraining

        Args:
            layer: quantized layer

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes

        """

        old_threshold = self.quantizer_parameters[self.PTQ_THRESHOLD]
        if self.quantization_parameter_learning:
            scale = tf.reshape(self.quantizer_parameters[self.SCALE_PTQ], self.threshold_shape)
            old_threshold = old_threshold * scale
        old_threshold = old_threshold.numpy()
        old_threshold = old_threshold.reshape(self.threshold_shape)
        return {THRESHOLD: old_threshold}
