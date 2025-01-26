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
import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, SiLU, Sigmoid, Linear, Hardtanh
from torch.nn.functional import relu, relu6

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.core.common.quantization.quantization_config import CustomOpsetLayers
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework import LayerFilterParams
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import \
    get_op_quantization_configs
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_configs
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

import model_compression_toolkit as mct



class BaseLayerFusingTest(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test=unit_test)
        self.expected_fusions = []

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 16, 16]]

    def representative_data_gen(self):
        input_shapes = self.create_inputs_shape()
        yield self.generate_inputs(input_shapes)

    def get_type(self, fusion):
        fusion_types = [x.type for x in fusion]
        return fusion_types

    def _compare(self, fused_nodes):
        self.unit_test.assertTrue(len(fused_nodes) == len(self.expected_fusions),
                                  msg=f'Number of fusions is not as expected!')
        for i, fusion in enumerate(fused_nodes):
            self.unit_test.assertTrue(self.get_type(fusion) == self.expected_fusions[i],
                                      msg=f'Miss-match fusion compared to expected!')


class LayerFusingTest1(BaseLayerFusingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_fusions = [[nn.Conv2d, nn.ReLU]]
        self.attach2fw = AttachTpcToPytorch()

    def get_tpc(self):
        base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
        default_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([default_config]))
        mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(mixed_precision_cfg_list),
                                                                                 base_config=base_config)
        conv = schema.OperatorsSet(name="Conv", qc_options=mixed_precision_configuration_options)
        any_relu = schema.OperatorsSet(name="ReLU")
        operator_set = [conv, any_relu]
        # Define fusions
        fusing_patterns = [schema.Fusing(operator_groups=(conv, any_relu))]
        generated_tp = schema.TargetPlatformCapabilities(default_qco=default_configuration_options,
                                                         tpc_minor_version=None,
                                                         tpc_patch_version=None,
                                                         tpc_platform_type=None,
                                                         operator_set=tuple(operator_set),
                                                         fusing_patterns=tuple(fusing_patterns),
                                                         name='layer_fusing_test')

        return generated_tp

    def run_test(self, seed=0):
        model_float = self.LayerFusingNetTest()

        graph = prepare_graph_with_configs(model_float, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                           self.representative_data_gen, lambda name, _tp: self.get_tpc(),
                                           attach2fw=self.attach2fw)

        self._compare(graph.fused_nodes)

    class LayerFusingNetTest(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3))
            self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 1))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            y = self.relu(x)
            return y


class LayerFusingTest2(BaseLayerFusingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_fusions = [[Conv2d, Hardtanh], [Conv2d, ReLU], [Conv2d, Sigmoid], [Conv2d, SiLU]]

    def get_tpc(self):
        base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
        mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(mixed_precision_cfg_list),
                                                                                 base_config=base_config)
        default_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([default_config]))
        conv = schema.OperatorsSet(name="Conv", qc_options=mixed_precision_configuration_options)
        any_act = schema.OperatorsSet(name="AnyAct")
        operator_set = [conv, any_act]
        # Define fusions
        fusing_patterns = [schema.Fusing(operator_groups=(conv, any_act))]
        generated_tp = schema.TargetPlatformCapabilities(default_qco=default_configuration_options,
                                                         tpc_minor_version=None,
                                                         tpc_patch_version=None,
                                                         tpc_platform_type=None,
                                                         operator_set=tuple(operator_set),
                                                         fusing_patterns=tuple(fusing_patterns),
                                                         name='layer_fusing_test')

        return generated_tp

    def run_test(self, seed=0):
        model_float = self.LayerFusingNetTest()
        graph = prepare_graph_with_configs(model_float, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                           self.representative_data_gen, lambda name, _tp: self.get_tpc(),
                                           attach2fw=AttachTpcToPytorch(),
                                           qc=QuantizationConfig(
                                               custom_tpc_opset_to_layer={"AnyAct": CustomOpsetLayers([ReLU, relu6, relu, SiLU, Sigmoid,
                                                                                      LayerFilterParams(Hardtanh, min_val=0)])}))

        self._compare(graph.fused_nodes)

    class LayerFusingNetTest(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
            self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 1))
            self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3))
            self.conv4 = nn.Conv2d(32, 64, kernel_size=(1, 1))
            self.conv5 = nn.Conv2d(64, 64, kernel_size=(2, 2))
            self.relu = nn.ReLU()
            self.tanh = Hardtanh(min_val=0)
            self.swish = nn.SiLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.tanh(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.sigmoid(x)
            x = self.conv5(x)
            y = self.swish(x)
            return y


class LayerFusingTest3(BaseLayerFusingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_fusions = [[Conv2d, ReLU]]

    def get_tpc(self):
        base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
        mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(mixed_precision_cfg_list),
                                                                                 base_config=base_config)
        default_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([default_config]))
        conv = schema.OperatorsSet(name="Conv", qc_options=mixed_precision_configuration_options)
        any_act = schema.OperatorsSet(name="AnyAct")
        operator_set = [conv, any_act]
        # Define fusions
        fusing_patterns = [schema.Fusing(operator_groups=(conv, any_act))]
        generated_tp = schema.TargetPlatformCapabilities(default_qco=default_configuration_options,
                                                         tpc_minor_version=None,
                                                         tpc_patch_version=None,
                                                         tpc_platform_type=None,
                                                         operator_set=tuple(operator_set),
                                                         fusing_patterns=tuple(fusing_patterns),
                                                         name='layer_fusing_test')
        return generated_tp

    def run_test(self, seed=0):
        model_float = self.LayerFusingNetTest()
        graph = prepare_graph_with_configs(model_float, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                           self.representative_data_gen, lambda name, _tp: self.get_tpc(),
                                           attach2fw=AttachTpcToPytorch(),
                                           qc=QuantizationConfig(
                                               custom_tpc_opset_to_layer={"AnyAct": CustomOpsetLayers([ReLU, relu6, relu])}))

        self._compare(graph.fused_nodes)

    class LayerFusingNetTest(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
            self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 1))
            self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3))
            self.conv4 = nn.Conv2d(32, 64, kernel_size=(1, 1))
            self.conv5 = nn.Conv2d(64, 64, kernel_size=(2, 2))
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()
            self.swish = nn.SiLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.tanh(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.sigmoid(x)
            x = self.conv5(x)
            y = self.swish(x)
            return y


class LayerFusingTest4(BaseLayerFusingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_fusions = [[Conv2d, SiLU, torch.add], [Conv2d, SiLU, torch.add], [Conv2d, ReLU],
                                 [Conv2d, ReLU, torch.add], [Linear, SiLU], [Linear, SiLU]]

    def get_tpc(self):
        base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
        mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(mixed_precision_cfg_list),
                                                                                 base_config=base_config)
        default_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([default_config]))
        conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=mixed_precision_configuration_options)
        fc = schema.OperatorsSet(name=schema.OperatorSetNames.FULLY_CONNECTED, qc_options=mixed_precision_configuration_options)
        relu = schema.OperatorsSet(name=schema.OperatorSetNames.RELU)
        add = schema.OperatorsSet(name=schema.OperatorSetNames.ADD)
        swish = schema.OperatorsSet(name=schema.OperatorSetNames.SWISH)
        operator_set = [conv, fc, relu, add, swish]
        activations_to_fuse = schema.OperatorSetGroup(operators_set=[relu, swish])
        # Define fusions
        fusing_patterns = [schema.Fusing(operator_groups=(conv, activations_to_fuse)),
                           schema.Fusing(operator_groups=(conv, add, activations_to_fuse)),
                           schema.Fusing(operator_groups=(conv, activations_to_fuse, add)),
                           schema.Fusing(operator_groups=(fc, activations_to_fuse))]

        generated_tp = schema.TargetPlatformCapabilities(default_qco=default_configuration_options,
                                                         tpc_minor_version=None,
                                                         tpc_patch_version=None,
                                                         tpc_platform_type=None,
                                                         operator_set=tuple(operator_set),
                                                         fusing_patterns=tuple(fusing_patterns),
                                                         name='layer_fusing_test')

        return generated_tp

    def run_test(self, seed=0):
        model_float = self.LayerFusingNetTest()
        graph = prepare_graph_with_configs(model_float, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                           self.representative_data_gen, lambda name, _tp: self.get_tpc(),
                                           attach2fw=AttachTpcToPytorch())

        self._compare(graph.fused_nodes)

    class LayerFusingNetTest(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding='same')
            self.conv2 = nn.Conv2d(3, 3, kernel_size=(1, 1), padding='same')
            self.conv3 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding='same')
            self.conv4 = nn.Conv2d(3, 3, kernel_size=(1, 1), padding='same')
            self.conv5 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding='same')
            self.conv6 = nn.Conv2d(3, 3, kernel_size=(1, 1), padding='same')
            self.relu = nn.ReLU()
            self.swish = nn.SiLU()
            self.flatten = nn.Flatten()
            self.dense1 = nn.Linear(768, out_features=16)
            self.dense2 = nn.Linear(16, out_features=16)

        def forward(self, inputs):
            x = self.conv1(inputs)
            x = self.swish(x)
            x1 = torch.add(inputs, x)
            x2 = self.conv2(x1)
            x2 = self.swish(x2)
            x2 = torch.add(x1, x2)
            x2 = self.conv3(x2)
            x2 = self.relu(x2)
            x3 = self.conv4(x2)
            x3 = self.relu(x3)
            x3 = torch.add(x3, x2)
            x3 = self.flatten(x3)
            x3 = self.dense1(x3)
            x3 = self.swish(x3)
            x3 = self.dense2(x3)
            y = self.swish(x3)
            return y
