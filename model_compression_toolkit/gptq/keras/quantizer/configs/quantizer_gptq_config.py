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

from typing import List, Tuple, Any, Dict

from tensorflow import Tensor
import tensorflow as tf

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.gptq.keras.quantizer.activation_quantizer.gptq_activation_quantizer import \
    GPTQActivationQuantizer

if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers import Layer
else:
    from keras.engine.base_layer import Layer

from tensorflow.python.training.tracking.data_structures import ListWrapper
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer

from model_compression_toolkit.gptq.keras.quantizer.configs.base_quantizer_gptq_config import BaseQuantizeConfig
from model_compression_toolkit.core.keras.constants import KERNEL

from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig, RoundingType
from model_compression_toolkit.gptq.keras.quantizer.gumbel_rounding.symmetric_gumbel import SymmetricGumbelRounding
from model_compression_toolkit.gptq.keras.quantizer.gumbel_rounding.uniform_gumbel import UniformGumbelRounding
from model_compression_toolkit.gptq.keras.quantizer.ste_rounding.symmetric_ste import STEWeightQuantizer
from model_compression_toolkit.core.common.target_platform.op_quantization_config import QuantizationMethod
from model_compression_toolkit.core.common.constants import THRESHOLD, RANGE_MAX, RANGE_MIN, SIGNED
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig, \
    NodeActivationQuantizationConfig
from model_compression_toolkit.gptq.common import gptq_constants


class GradientPTQQuantizeConfig(BaseQuantizeConfig):
    """
    QuantizeConfig to quantize the weights of a layer using a TrainableQuantizer.
    """

    def __init__(self,
                 weight_attrs: List[str],
                 final_weights_quantization_cfg: NodeWeightsQuantizationConfig,
                 final_activation_quantization_cfg: NodeActivationQuantizationConfig,
                 gptq_config: GradientPTQConfig):
        """
        Initialize a TrainableQuantizer and sets the weights and activation quantizers,
        based on the provided config.

        Args:
            weight_attrs: Attributes of the layer's weights to quantize.
            final_weights_quantization_cfg: quantization config of the current layer.
            final_activation_quantization_cfg: NodeActivationQuantizationConfig,
            gptq_config: A GPTQ configuration calls.
        """

        self.weight_attrs = weight_attrs
        self.final_weights_quantization_cfg = final_weights_quantization_cfg
        self.final_activation_quantization_cfg = final_activation_quantization_cfg
        self.gptq_config = gptq_config

        if final_weights_quantization_cfg.enable_weights_quantization:
            self._build_weights_quantizer()
        else:
            self.weight_quantizer = None

        if final_activation_quantization_cfg.enable_activation_quantization:
            self._build_activation_quantizer()
        else:
            self.activation_quantizer = None

    def _build_weights_quantizer(self):
        num_bits = self.final_weights_quantization_cfg.weights_n_bits
        weight_channel_axis = self.final_weights_quantization_cfg.weights_channels_axis
        max_lsbs_change_map = self.gptq_config.lsb_change_per_bit_width
        if self.final_weights_quantization_cfg.weights_quantization_method in [QuantizationMethod.SYMMETRIC,
                                                                               QuantizationMethod.POWER_OF_TWO]:
            is_power_of_two = QuantizationMethod.POWER_OF_TWO == self.final_weights_quantization_cfg.weights_quantization_method
            weights_threshold_values = self.final_weights_quantization_cfg.weights_quantization_params.get(THRESHOLD)
            if self.gptq_config.rounding_type == RoundingType.STE:
                self.weight_quantizer = STEWeightQuantizer(num_bits=num_bits,
                                                           per_axis=len(
                                                               weights_threshold_values.flatten()) > 1,
                                                           threshold_values=weights_threshold_values,
                                                           signed=True,
                                                           power_of_two=is_power_of_two,
                                                           quantization_axis=weight_channel_axis,
                                                           max_lsbs_change_map=max_lsbs_change_map)
            elif self.gptq_config.rounding_type == RoundingType.GumbelRounding:
                self.weight_quantizer = SymmetricGumbelRounding(num_bits=num_bits,
                                                                per_axis=len(
                                                                    weights_threshold_values.flatten()) > 1,
                                                                threshold_values=weights_threshold_values,
                                                                signed=True,
                                                                power_of_two=is_power_of_two,
                                                                quantization_parameter_learning=self.gptq_config.quantization_parameters_learning,
                                                                quantization_axis=weight_channel_axis,
                                                                max_lsbs_change_map=max_lsbs_change_map,
                                                                max_iteration=self.gptq_config.n_iter,
                                                                gumbel_config=self.gptq_config.quantizer_config)
            else:
                common.Logger.error(
                    f"For quantization method {self.final_weights_quantization_cfg.weights_quantization_method}, GPTQ Rounding type {self.gptq_config.rounding_type} is not supported")
        elif self.final_weights_quantization_cfg.weights_quantization_method == QuantizationMethod.UNIFORM:
            if not self.gptq_config.rounding_type == RoundingType.GumbelRounding:
                common.Logger.error(
                    f"For quantization method {self.final_weights_quantization_cfg.weights_quantization_method}, GPTQ Rounding type {self.gptq_config.rounding_type} is not supported")
            range_max = self.final_weights_quantization_cfg.weights_quantization_params.get(RANGE_MAX)
            range_min = self.final_weights_quantization_cfg.weights_quantization_params.get(RANGE_MIN)
            self.weight_quantizer = UniformGumbelRounding(num_bits=num_bits,
                                                          per_axis=len(
                                                              range_max.flatten()) > 1,
                                                          min_range=range_min,
                                                          max_range=range_max,
                                                          signed=True,
                                                          quantization_parameter_learning=self.gptq_config.quantization_parameters_learning,
                                                          quantization_axis=weight_channel_axis,
                                                          max_lsbs_change_map=max_lsbs_change_map,
                                                          max_iteration=self.gptq_config.n_iter,
                                                          gumbel_config=self.gptq_config.quantizer_config)

    def _build_activation_quantizer(self):
        if self.final_activation_quantization_cfg.activation_quantization_method in [QuantizationMethod.SYMMETRIC]:
            activation_threshold_value = self.final_activation_quantization_cfg.activation_quantization_params.get(THRESHOLD)
            num_bits = self.final_activation_quantization_cfg.activation_n_bits
            signed = self.final_activation_quantization_cfg.activation_quantization_params.get(SIGNED)

            self.activation_quantizer = GPTQActivationQuantizer(num_bits=num_bits,
                                                                signed=signed,
                                                                threshold_value=activation_threshold_value)
        else:
            raise NotImplementedError("Activation GPTQ is currently supported only for Symmetric quantization")

    def enable_update(self):
        """
        This function enable the parameter update (update iteration index and gumbel random variable)
        Returns: None

        """
        if self.gptq_config.is_gumbel:
            return self.weight_quantizer.enable_update()

    def disable_update(self):
        """

        This function disable the parameter update (update iteration index and gumbel random variable)
        Returns: None

        """
        if self.gptq_config.is_gumbel:
            return self.weight_quantizer.disable_update()

    def get_weights_and_quantizers(self, layer: Layer) -> List[Tuple[Tensor, Quantizer]]:
        """
        Get a list of tuples with weights and the weight quantizer.
        The layer's attributes are used to get the weights.
        Args:
            layer: The layer the WeightQuantizeConfig wraps.

        Returns:
            List of tuples of the layer's weights and the weight quantizer.
        """
        if self.final_weights_quantization_cfg.enable_weights_quantization:
            return [(getattr(layer, weight_attr), self.weight_quantizer)
                    for weight_attr in self.weight_attrs]
        else:
            return []

    def get_activations_and_quantizers(self, layer: Layer) -> list:
        # For activation quantization parameters training we use get_output_quantizers,
        # Therefore, we do not need to implement this method.
        return []

    def set_quantize_weights(self, layer: Layer, quantize_weights: List[Tensor]):
        """
        Set the layer weights with new passed weights.
        Args:
            layer: Layer to set its attributes.
            quantize_weights: Quantized weights to set as new weights.

        """
        if self.final_weights_quantization_cfg.enable_weights_quantization:
            if len(self.weight_attrs) != len(quantize_weights):
                raise ValueError(
                    '`set_quantize_weights` called on layer {} with {} '
                    'weight parameters, but layer expects {} values.'.format(
                        layer.name, len(quantize_weights), len(self.weight_attrs)))  # pragma: no cover

            for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
                current_weight = getattr(layer, weight_attr)
                if current_weight.shape != weight.shape:
                    raise ValueError('Existing layer weight shape {} is incompatible with'
                                     'provided weight shape {}'.format(
                        current_weight.shape, weight.shape))  # pragma: no cover

                setattr(layer, weight_attr, weight)

    def set_quantize_activations(self, layer, quantize_activations: ListWrapper):
        pass

    def get_output_quantizers(self, layer: Layer) -> list:
        return \
            [] if not self.final_activation_quantization_cfg.enable_activation_quantization \
            else [self.activation_quantizer]

    @classmethod
    def from_config(cls, config: dict):
        """
        Instantiates a `WeightQuantizeConfig` from its config.

        Args:
            config: Output of `get_config()`.

        Returns:
            A `WeightQuantizeConfig` instance.
        """

        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns: The WeightQuantizeConfig configuration.
        """
        return {
            'weight_attrs': self.weight_attrs,
            'final_weights_quantization_cfg': self.final_weights_quantization_cfg,
            'final_activation_quantization_cfg': self.final_activation_quantization_cfg,
            'gptq_config': self.gptq_config,
        }

    def update_layer_quantization_params(self, layer: Layer) -> (Dict[str, tf.Tensor], Dict[str, Dict], Dict):
        """
        A Function to calculate the needed change in attributes in NodeQuantizationConfig after retraining.
        Usually a function of the config quantizers.

        Args:
            layer: layer being quantized.

        Returns:
            3 dictionaries describing the change in layer's weights, weights config, activation config
            that changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes

        """
        weights = {}
        weights_quant_config = {}
        activation_quant_config = {}

        if self.final_weights_quantization_cfg.enable_weights_quantization:
            for weight, quantizer, quantizer_vars in layer._weight_vars:
                weights.update({KERNEL: quantizer(weight, training=False, weights=quantizer_vars)})

            weights_quant_config = {gptq_constants.WEIGHTS_QUANTIZATION_PARAMS: self.weight_quantizer.get_quant_config(layer)}

        if self.final_activation_quantization_cfg.enable_activation_quantization:
            activation_quant_config = {gptq_constants.ACTIVATION_QUANTIZATION_PARAMS: self.activation_quantizer.get_quant_config(layer)}

        return weights, weights_quant_config, activation_quant_config

    def get_trainable_quantizer_parameters(self) -> List[tf.Tensor]:
        """
        A function to get a list trainable of trainable parameters for GPTQ retraining from config quantizers

        Returns:
            A list of trainable Tensors

        """
        return self.weight_quantizer.get_trainable_parameters()

    def get_aux_variable(self) -> List[tf.Tensor]:
        return [self.weight_quantizer.get_aux_variable()]

    def get_weights_quantization_variable(self) -> List[tf.Tensor]:
        return self.weight_quantizer.get_quantization_variable()

    def get_activation_quantization_variable(self) -> List[tf.Tensor]:
        return self.activation_quantizer.get_quantization_variable()

    def get_temperature_variable(self) -> tf.Tensor:
        if self.gptq_config.is_gumbel:
            return self.weight_quantizer.get_temperature_variable()
        else:
            common.logger.Logger.error("Temperature variable only exist when using Gumbel Rounding Quantizer")

    def get_gumbel_probability(self) -> tf.Tensor:
        if self.gptq_config.is_gumbel:
            return self.weight_quantizer.get_gumbel_probability()
        else:
            common.logger.Logger.error("Probability variable only exist when using Gumbel Rounding Quantizer")

    def __eq__(self, other: Any) -> bool:
        """
        Check whether it equals to another object or not.
        """
        if not isinstance(other, GradientPTQQuantizeConfig):
            return False

        return (self.weight_attrs == other.weight_attrs and
                self.weight_quantizer == other.weight_quantizer and
                self.activation_quantizer == other.activation_quantizer and
                self.gptq_config == other.gptq_config)

    def __ne__(self, other: Any) -> bool:
        """
        Check whether it differs from another object or not.
        """
        return not self.__eq__(other)
