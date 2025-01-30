from unittest.mock import Mock
import networkx as nx

import pytest
import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU, Linear, Hardswish

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian import HessianMode, HessianScoresGranularity, HessianScoresRequest
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.graph_prep_runner import read_model_to_graph
from model_compression_toolkit.core.pytorch.constants import KERNEL
from model_compression_toolkit.core.pytorch.data_util import data_gen_to_dataloader
import numpy as np

from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.hessian.activation_hessian_scores_calculator_pytorch import \
    ActivationHessianScoresCalculatorPytorch
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_configs
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

BATCH_SIZE = 4


@pytest.fixture
def x():
    return 1 + torch.randn((BATCH_SIZE, 3, 16, 16), generator=torch.Generator().manual_seed(42)).to(device=get_working_device())


class BasicModel(torch.nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        torch.manual_seed(42)
        self.conv1 = Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.bn = BatchNorm2d(3)
        self.relu = ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TestActivationHessianScoresCalculator:

    def test_per_tensor_hessian_computation(self, x):
        print(test_per_tensor_hessian_computation)
        def representative_data_gen():
            yield [x]

        mock_fqc = Mock()
        mock_fqc.op_sets_to_layers.get_layers.return_value = []

        graph = read_model_to_graph(in_model=BasicModel(),
                                    representative_data_gen=representative_data_gen,
                                    fqc=mock_fqc,
                                    fw_info=Mock(),
                                    fw_impl=PytorchImplementation())
        
        # candidate_mock = Mock(spec=CandidateNodeQuantizationConfig)
        # candidate_mock.ena
        # candidate_mock.
        # TODO: use mocks correctly
        # TODO: extract graph prep to base
        graph.get_configurable_sorted_nodes_names = lambda aa: False
        for n in graph.nodes:
        #     n.candidates_quantization_cfg = []
            n.is_activation_quantization_enabled = lambda: False

        #
        # expected_order = list(nx.topological_sort(graph))
        # def mock_topological_sort(g):
        #     yield from expected_order
        #
        # mock_graph = Mock(wraps=graph)
        # mock_graph.get_configurable_sorted_nodes_names.return_value = []
        # mock_graph.topological_sort = mock_topological_sort

        target_nodes = [n for n in graph.get_topo_sorted_nodes() if n.name == 'conv1']
        request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                       granularity=HessianScoresGranularity.PER_TENSOR,
                                       target_nodes=target_nodes,
                                       data_loader=None,  # irrelevant for the test
                                       n_samples=Mock()  # irrelevant for the test
        )

        hessian_scores_calculator = ActivationHessianScoresCalculatorPytorch(
            graph,
            input_images=[x],
            fw_impl=PytorchImplementation(),
            hessian_scores_request=request,
            num_iterations_for_approximation=20,
        )

        res = hessian_scores_calculator.compute()

        assert len(res) == len(target_nodes)
        # for node_hess_score in res:
        #     assert node_hess_score.shape == (BATCH_SIZE, 1)

        expected_hess_results = np.array([499, 508, 497, 499]).reshape((BATCH_SIZE, 1))
        assert np.allclose(res[0], expected_hess_results, atol=0.01, rtol=0.0)

        # TODO: make setting for a general test for activation (extract expected results, model, target nodes and iterate over nodes to assert results
        # TODO: is there a better way to get expected constant results?
        # TODO: extend to all activation tests


        #
        # expected_shape = (num_scores, 1)
        # for node_scores in info.values():
        #     self.unit_test.assertTrue(node_scores.shape[0] == num_scores,
        #                               f"Requested {num_scores} score but {node_scores.shape[0]} scores were fetched")
        #
        #     self.unit_test.assertTrue(node_scores.shape == expected_shape,
        #                               f"Tensor shape is expected to be {expected_shape} but has shape {node_scores.shape}")