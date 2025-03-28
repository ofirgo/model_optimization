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
import torch.nn
from model_compression_toolkit.core import MixedPrecisionQuantizationConfig, ResourceUtilization, MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.pytorch.reader.node_holders import DummyPlaceHolder
from model_compression_toolkit.core.common.quantization.quantization_config import CustomOpsetLayers
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs
from tests.common_tests.helpers.generate_test_tpc import generate_tpc_with_activation_mp
from tests.pytorch_tests.tpc_pytorch import get_mp_activation_pytorch_tpc_dict

import model_compression_toolkit as mct

def get_base_mp_nbits_candidates():
    return [(4, 8), (4, 4), (4, 2),
            (8, 8), (8, 4), (8, 2),
            (2, 8), (2, 4), (2, 2)]


class BaseBopsNetwork(torch.nn.Module):
    def __init__(self, input_shape):
        super(BaseBopsNetwork, self).__init__()

        _, in_channels, _, _ = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, 4, kernel_size=(3, 3))
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(4, 4, kernel_size=(3, 3))

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.relu(x)
        output = self.conv2(x)

        return output


class MultipleEdgesBopsNetwork(torch.nn.Module):
    def __init__(self, input_shape):
        super(MultipleEdgesBopsNetwork, self).__init__()

        _, in_channels, _, _ = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, 4, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(in_channels, 4, kernel_size=(3, 3))
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        y = self.conv2(inp)

        x = self.relu1(x)
        y = self.relu2(y)

        output = x + y
        return output


class AllWeightsBopsNetwork(torch.nn.Module):
    def __init__(self, input_shape):
        super(AllWeightsBopsNetwork, self).__init__()

        _, in_channels, _, _ = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, 4, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(4, 4, kernel_size=(3, 3))
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.relu1 = torch.nn.ReLU()
        self.conv_trans = torch.nn.ConvTranspose2d(4, 4, kernel_size=(3, 3))
        self.bn2 = torch.nn.BatchNorm2d(4)
        self.relu2 = torch.nn.ReLU()
        self.depthwise = torch.nn.Conv2d(4, 4, kernel_size=(1, 1), groups=4)
        self.linear = torch.nn.Linear(30, 5)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv_trans(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.depthwise(x)
        output = self.linear(x)

        return output


class BaseMixedPrecisionBopsTest(BasePytorchTest):
    def __init__(self, unit_test, mixed_precision_candidates_list):
        super().__init__(unit_test)

        self.mixed_precision_candidates_list = mixed_precision_candidates_list

    def get_tpc(self):
        base_config, _, default_config = get_op_quantization_configs()
        return get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tpc_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=self.mixed_precision_candidates_list),
            test_name='mixed_precision_bops_model',
            tpc_name='mixed_precision_bops_pytorch_test')

    def get_core_configs(self):
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                         relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                         input_scaling=False, activation_channel_equalization=False,
                                         custom_tpc_opset_to_layer={"Input": CustomOpsetLayers([DummyPlaceHolder])})
        mpc = MixedPrecisionQuantizationConfig(num_of_images=1)

        return {"mixed_precision_bops_model": mct.core.CoreConfig(quantization_config=qc, mixed_precision_config=mpc)}

    def get_input_shapes(self):
        return [[self.val_batch_size, 16, 16, 3]]

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # Verify that some layers got bit-width smaller than 8 bits (so checking candidate index is not 0)
        self.unit_test.assertTrue(any(i > 0 for i in quantization_info.mixed_precision_cfg))
        # Verify final BOPs utilization
        self.unit_test.assertTrue(self.get_resource_utilization().is_satisfied_by(quantization_info.final_resource_utilization))


class MixedPrecisionBopsBasicTest(BaseMixedPrecisionBopsTest):
    def __init__(self, unit_test):

        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        super().__init__(unit_test, mixed_precision_candidates_list)

    def create_feature_network(self, input_shape):
        return BaseBopsNetwork(input_shape)

    def get_resource_utilization(self):
        return ResourceUtilization(bops=1350000)  # should require some quantization to all layers


class MixedPrecisionBopsAllWeightsLayersTest(BaseMixedPrecisionBopsTest):
    def __init__(self, unit_test, mixed_precision_candidates_list=None):

        if mixed_precision_candidates_list is None:
            mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        super().__init__(unit_test, mixed_precision_candidates_list)

    def create_feature_network(self, input_shape):
        return AllWeightsBopsNetwork(input_shape)

    def get_resource_utilization(self):
        return ResourceUtilization(bops=3000000)  # should require some quantization to all layers


class MixedPrecisionWeightsOnlyBopsTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, mixed_precision_candidates_list=[(8, 8), (4, 8), (2, 8)])

    def get_resource_utilization(self):
        return ResourceUtilization(bops=10000000)  # should require some quantization to all layers


class MixedPrecisionActivationOnlyBopsTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, mixed_precision_candidates_list=[(8, 8), (8, 4), (8, 2)])

    def get_resource_utilization(self):
        return ResourceUtilization(bops=10000000)  # should require some quantization to all layers

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # Verify that some layers got bit-width smaller than 8 bits (so checking candidate index is not 0)
        self.unit_test.assertTrue(any(i > 0 for i in quantization_info.mixed_precision_cfg))
        # Verify final BOPs utilization
        self.unit_test.assertTrue(quantization_info.final_resource_utilization.bops <= self.get_resource_utilization().bops)


class MixedPrecisionBopsAndWeightsMemoryUtilizationTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        return ResourceUtilization(weights_memory=150, bops=3000000)  # should require some quantization to all layers


class MixedPrecisionBopsAndActivationMemoryUtilizationTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        # act max cut 2bit = 2*(30*30*4)*2/8 = 1800
        return ResourceUtilization(activation_memory=1800, bops=3000000)  # should require some quantization to all layers


class MixedPrecisionBopsAndTotalMemoryUtilizationTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        return ResourceUtilization(total_memory=2000, bops=3000000)  # should require some quantization to all layers


class MixedPrecisionBopsWeightsActivationUtilizationTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        return ResourceUtilization(weights_memory=150, activation_memory=1800, bops=3000000)  # should require some quantization to all layers


class MixedPrecisionBopsMultipleOutEdgesTest(BaseMixedPrecisionBopsTest):
    def __init__(self, unit_test):

        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        super().__init__(unit_test, mixed_precision_candidates_list)

    def create_feature_network(self, input_shape):
        return MultipleEdgesBopsNetwork(input_shape)

    def get_resource_utilization(self):
        return ResourceUtilization(bops=1)  # No layers with BOPs count

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # Verify that all layers got 8 bits (so checking candidate index is 0)
        self.unit_test.assertTrue(all(i == 0 for i in quantization_info.mixed_precision_cfg))