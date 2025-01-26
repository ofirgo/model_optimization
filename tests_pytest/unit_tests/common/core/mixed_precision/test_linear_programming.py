# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

from unittest.mock import MagicMock, patch

import pytest
import numpy as np
from pulp import lpSum

from model_compression_toolkit.constants import MIN_THRESHOLD, THRESHOLD, SIGNED
from model_compression_toolkit.core import QuantizationErrorMethod, ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget
from model_compression_toolkit.core.common.mixed_precision.search_methods.linear_programming import \
    mp_integer_programming_search
from model_compression_toolkit.core.common.quantization.quantization_params_generation.symmetric_selection import \
    symmetric_selection_histogram


class TestMixedPrecisionLinearProgrammingSolver:

    def _mock_compute_function(self, ru_sum_vector):
        return [lpSum((ru_sum_vector))]

    def _mock_reconstruct_config_from_virtual_graph(self, x, *args, **kwargs):
        return x

    @pytest.mark.parametrize("target_resource_utilization, tested_targets", [(ResourceUtilization(weights_memory=2), [RUTarget.WEIGHTS]),
                                                                            (ResourceUtilization(activation_memory=2), [RUTarget.ACTIVATION]),
                                                                            (ResourceUtilization(total_memory=2), [RUTarget.TOTAL]),
                                                                            (ResourceUtilization(weights_memory=2, activation_memory=2),
                                                                             [RUTarget.WEIGHTS, RUTarget.ACTIVATION]),
                                                                             (ResourceUtilization(bops=2), [RUTarget.BOPS])])
    def test_linear_programming_solver(self, target_resource_utilization, tested_targets):
        _num_configurable_layers = 3

        search_manager_mock = MagicMock()
        search_manager_mock.max_ru_config = [0 for _ in range(_num_configurable_layers)]
        search_manager_mock.config_reconstruction_helper.reconstruct_config_from_virtual_graph = self._mock_reconstruct_config_from_virtual_graph
        search_manager_mock.layer_to_bitwidth_mapping = {i: [0, 1, 2] for i in range(_num_configurable_layers)}
        search_manager_mock.min_ru = {RUTarget.WEIGHTS: [[0.25] for _ in range(_num_configurable_layers)],
                                      RUTarget.ACTIVATION: [[0.25] for _ in range(_num_configurable_layers)],
                                      RUTarget.TOTAL: [[0.25] for _ in range(_num_configurable_layers)],
                                      RUTarget.BOPS: [[0.25] for _ in range(_num_configurable_layers)]}

        search_manager_mock.compute_ru_functions = {tt: [None, self._mock_compute_function] for tt in tested_targets}
        search_manager_mock.non_conf_ru_dict = {tt: np.array([]) for tt in tested_targets}

        # Setting ordered return values for each call to 'search_manager.compute_metric_fn' during the execution of
        # the tested function.
        # Config calls order - [0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 1], [0, 0, 2]
        # The expected metric mapping output:
        # {0: {0: 1, 1: 2, 2: 2.5}, 1: {0: 1, 1: 3, 2: 3.5}, 2: {0: 1, 1: 2.25, 2: 2.4}}
        search_manager_mock.compute_metric_fn.side_effect = [1, 2, 2.5, 3, 3.5, 2.25, 2.4]

        search_manager_mock.compute_resource_utilization_matrix.return_value = np.array([[1, 0.5, 0.25, 1, 0.5, 0.25, 1, 0.5, 0.25]]).T

        mp_res = mp_integer_programming_search(search_manager_mock, target_resource_utilization)

        assert all([mp_res[i] == [1, 0, 2][i] for i in range(_num_configurable_layers)])
        # TODO: extend the verifications, can we check the values of arguments to inner function calls?
        # TODO: add test with non configurable nodes

        # with patch.object(mp_integer_programming_search, '_formalize_problem') as mock_formalize_problem:
        #     mock_formalize_problem.side_effect = lambda *args, **kwargs: args  # Capture the arguments
        #     mp_res = mp_integer_programming_search(search_manager_mock, target_resource_utilization)
        #
        #     assert mock_formalize_problem.call_count > 0
        #     for call_args in mock_formalize_problem.call_args_list:
        #         assert call_args is not None




