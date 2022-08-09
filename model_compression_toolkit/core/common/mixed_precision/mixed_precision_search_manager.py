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

from typing import Callable, Tuple
from typing import Dict, List
import numpy as np

from model_compression_toolkit.core.common.constants import VIRTUAL_ACTIVATION_SUFFIX, VIRTUAL_WEIGHTS_SUFFIX
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualActivationWeightsNode, \
    VirtualSplitWeightsNode, VirtualSplitActivationNode
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPITarget
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_aggregation_methods import MpKpiAggregation
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_methods import MpKpiMetric
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.sensitivity_evaluation import SensitivityEvaluation


class MixedPrecisionSearchManager:
    """
    Class to wrap and manage the search process of a mixed-precision configuration.
    """

    def __init__(self,
                 graph: Graph,
                 fw_info: FrameworkInfo,
                 fw_impl: FrameworkImplementation,
                 sensitivity_evaluator: SensitivityEvaluation,
                 kpi_functions: Dict[KPITarget, Tuple[MpKpiMetric, MpKpiAggregation]],
                 original_graph: Graph = None):
        """

        Args:
            graph: Graph to search for its MP configuration.
            fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
            sensitivity_evaluator: A SensitivityEvaluation which provides a function that evaluates the sensitivity of
                a bit-width configuration for the MP model.
            kpi_functions: A dictionary with pairs of (MpKpiMethod, MpKpiAggregationMethod) mapping a KPITarget to
                a couple of kpi metric function and kpi aggregation function.
        """

        self.graph = graph
        self.original_graph = graph if original_graph is None else original_graph
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.sensitivity_evaluator = sensitivity_evaluator
        self.layer_to_bitwidth_mapping = self.get_search_space()
        self.compute_metric_fn = self.get_sensitivity_metric()

        self.compute_kpi_functions = kpi_functions

        self.min_kpi_config = self.graph.get_min_candidates_config()
        self.max_kpi_config = self.graph.get_max_candidates_config()

        self.min_kpi = self.compute_min_kpis()

    def get_search_space(self) -> Dict[int, List[int]]:
        """
        The search space is a mapping from a node's index to a list of integers (possible bitwidths candidates indeces
        for the node).

        Returns:
            The entire search space of the graph.
        """

        indices_mapping = {}
        nodes_to_configure = self.graph.get_configurable_sorted_nodes()
        for idx, n in enumerate(nodes_to_configure):
            # For each node, get all possible bitwidth indices for it
            # (which is a list from 0 to the length of the candidates mp_config list of the node).
            indices_mapping[idx] = list(range(len(n.candidates_quantization_cfg)))  # all search_methods space
        return indices_mapping

    def get_sensitivity_metric(self) -> Callable:
        """

        Returns: Return a function (from the framework implementation) to compute a metric that
        indicates the similarity of the mixed-precision model (to the float model) for a given
        mixed-precision configuration.

        """
        # Get from the framework an evaluation function on how a MP configuration,
        # affects the expected loss.

        return self.sensitivity_evaluator.compute_metric

    def compute_min_kpis(self) -> Dict[KPITarget, np.ndarray]:
        """
        Computes a KPIs vector with the values matching to the minimal mp configuration
        (i.e., each node is configured with the quantization candidate that would give the minimal size of the
        node's KPI).
        The method computes the minimal KPIs vector for each kpi target.

        Returns: A dictionary mapping each kpi target to its respective minimal KPIs values.

        """
        min_kpis = {}
        for kpi_target, kpi_fns in self.compute_kpi_functions.items():
            # kpi_fns is a pair of kpi computation method and kpi aggregation method (in this method we only need
            # the first one)
            min_kpis[kpi_target] = kpi_fns[0](self.min_kpi_config, self.graph, self.fw_info, self.fw_impl)

        return min_kpis

    def compute_kpi_matrix(self, target: KPITarget) -> np.ndarray:
        """
        Computes and builds a KPIs matrix, to be used for the mixed-precision search problem formalization.
        The matrix is constructed as follows (for a given target):
        - Each row represents the set of KPI values for a specific KPI measure (number of rows should be equal to the
            length of the output of the respective target compute_kpi function).
        - Each entry in a specific column represents the KPI value of a given configuration (single layer is configured
            with specific candidate, all other layer are at the minimal KPI configuration) for the KPI measure of the
            respective row.

        Args:
            target: The target for which the KPI is calculated (a KPITarget value).

        Returns: A KPI matrix.

        """
        assert isinstance(target, KPITarget), f"{target} is not a valid KPI target"

        configurable_sorted_nodes = self.graph.get_configurable_sorted_nodes()

        kpi_matrix = []
        for c, c_n in enumerate(configurable_sorted_nodes):
            for candidate_idx in range(len(c_n.candidates_quantization_cfg)):
                candidate_kpis = self.compute_candidate_relative_kpis(c, candidate_idx, target)
                kpi_matrix.append(np.asarray(candidate_kpis))

        # We need to transpose the calculated kpi matrix to allow later multiplication with
        # the indicators' diagonal matrix.
        # We only move the first axis (num of configurations) to be last,
        # the remaining axes include the metric specific nodes (rows dimension of the new tensor)
        # and the kpi metric values (if they are non-scalars)
        np_kpi_matrix = np.array(kpi_matrix)
        return np.moveaxis(np_kpi_matrix, source=0, destination=len(np_kpi_matrix.shape) - 1)

    def compute_candidate_relative_kpis(self,
                                        conf_node_idx: int,
                                        candidate_idx: int,
                                        target: KPITarget) -> np.ndarray:
        """
        Computes a KPIs vector for a given candidates of a given configurable node, i.e., the matching KPI vector
        which is obtained by computing the given target's KPI function on a minimal configuration in which the given
        layer's candidates is changed to the new given one.
        The result is normalized by subtracting the target's minimal KPIs vector.

        Args:
            conf_node_idx: The index of a node in a sorted configurable nodes list.
            candidate_idx: The index of a node's quantization configuration candidate.
            target: The target for which the KPI is calculated (a KPITarget value).

        Returns: Normalized node's KPIs vector

        """
        return self.compute_node_kpi_for_candidate(conf_node_idx, candidate_idx, target) - \
               self.get_min_target_kpi(target)

    def get_min_target_kpi(self, target: KPITarget) -> np.ndarray:
        """
        Returns the minimal KPIs vector (pre-calculated on initialization) of a specific target.

        Args:
            target: The target for which the KPI is calculated (a KPITarget value).

        Returns: Minimal KPIs vector.

        """
        return self.min_kpi[target]

    def compute_node_kpi_for_candidate(self, conf_node_idx: int, candidate_idx: int, target: KPITarget) -> np.ndarray:
        """
        Computes a KPIs vector after replacing the given node's configuration candidate in the minimal
        target configuration with the given candidate index.

        Args:
            conf_node_idx: The index of a node in a sorted configurable nodes list.
            candidate_idx: Quantization config candidate to be used for the node's KPI computation.
            target: The target for which the KPI is calculated (a KPITarget value).

        Returns: Node's KPIs vector.

        """
        return self.compute_kpi_functions[target][0](
            self.replace_config_in_index(
                self.min_kpi_config,
                conf_node_idx,
                candidate_idx),
            self.graph,
            self.fw_info,
            self.fw_impl)

    @staticmethod
    def replace_config_in_index(mp_cfg: List[int], idx: int, value: int) -> List[int]:
        """
        Replacing the quantization configuration candidate in a given mixed-precision configuration at the given
        index (node's index) with the given value (candidate index).

        Args:
            mp_cfg: Mixed-precision configuration (list of candidates' indices)
            idx: A configurable node's index.
            value: A new candidate index to configure.

        Returns: A new mixed-precision configuration.

        """
        updated_cfg = mp_cfg.copy()
        updated_cfg[idx] = value
        return updated_cfg

    def weights_only(self, weights_node, virtual_node, origin_conf_nodes, virtual_cfg_idx, origin_node_idx_to_cfg):
        if weights_node.name in origin_conf_nodes:
            # It is possible that the original weights node is not configurable,
            # in this case we don't need to retrieve its bit-width config
            weights_bitwidth = virtual_node.candidates_quantization_cfg[virtual_cfg_idx].weights_quantization_cfg.weights_n_bits
            origin_cfg_idx = [i for i, c in
                              enumerate(weights_node.candidates_quantization_cfg) if
                              c.weights_quantization_cfg.weights_n_bits == weights_bitwidth]

            origin_idx = origin_conf_nodes.index(weights_node.name)
            origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx[0]

    def activation_only(self, activation_node, virtual_node, origin_conf_nodes, virtual_cfg_idx, origin_node_idx_to_cfg):
        if activation_node.name in origin_conf_nodes:
            # It is possible that the original activation node is not configurable,
            # in this case we don't need to retrieve its bit-width config
            activation_bitwidth = virtual_node.candidates_quantization_cfg[
                virtual_cfg_idx].activation_quantization_cfg.activation_n_bits
            origin_cfg_idx = [i for i, c in
                              enumerate(activation_node.candidates_quantization_cfg) if
                              c.activation_quantization_cfg.activation_n_bits == activation_bitwidth]

            origin_idx = origin_conf_nodes.index(activation_node.name)
            origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx[0]

    def activation_weights(self, activation_node, weights_node, virtual_node, origin_conf_nodes, virtual_cfg_idx, virtual_mp_cfg, virtual_sorted_nodes_names, origin_node_idx_to_cfg):
        activation_bitwidth = activation_node.candidates_quantization_cfg[virtual_mp_cfg[
            virtual_sorted_nodes_names.index(activation_node.name)]].activation_quantization_cfg.activation_n_bits
        weights_bitwidth = virtual_node.candidates_quantization_cfg[virtual_cfg_idx].weights_quantization_cfg.weights_n_bits
        origin_cfg_idx = [i for i, c in
                          enumerate(weights_node.origin_node.candidates_quantization_cfg) if
                          c.weights_quantization_cfg.weights_n_bits == weights_bitwidth and
                          c.activation_quantization_cfg.activation_n_bits == activation_bitwidth]

        origin_idx = origin_conf_nodes.index(weights_node.origin_node.name)
        origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx[0]

    def weights_activation(self, activation_node, weights_node, virtual_node, origin_conf_nodes, virtual_cfg_idx, virtual_mp_cfg, virtual_sorted_nodes_names, origin_node_idx_to_cfg):
        weights_bitwidth = weights_node.candidates_quantization_cfg[virtual_mp_cfg[virtual_sorted_nodes_names.index(weights_node.name)]].weights_quantization_cfg.weights_n_bits
        activation_bitwidth = virtual_node.candidates_quantization_cfg[
            virtual_cfg_idx].activation_quantization_cfg.activation_n_bits
        origin_cfg_idx = [i for i, c in enumerate(activation_node.origin_node.candidates_quantization_cfg) if
                          c.weights_quantization_cfg.weights_n_bits == weights_bitwidth and
                          c.activation_quantization_cfg.activation_n_bits == activation_bitwidth]

        if activation_node.origin_node.name not in origin_conf_nodes:
            # TODO: all nodes should be configurable in the original graph
            raise Exception()
        origin_idx = origin_conf_nodes.index(activation_node.origin_node.name)
        origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx[0]

    def get_activation_for_split_weights(self, weights_node, virtual_node, virtual_cfg_idx, virtual_sorted_nodes_names, virtual_mp_cfg, origin_conf_nodes, origin_node_idx_to_cfg):
        # This is a weights node that was split, means it has an activation node that should follow it,
        # and we need its configuration in order to reconstruct the original node's configuration.
        matching_activation_node = self.graph.get_next_nodes(virtual_node)
        assert len(matching_activation_node) == 1
        activation_node = matching_activation_node[0]

        if isinstance(activation_node, VirtualActivationWeightsNode):
            if activation_node.original_activation_node.is_activation_quantization_enabled() and not \
                    activation_node.original_activation_node.is_all_activation_candidates_equal():
                assert activation_node.name in virtual_sorted_nodes_names  # Sanity check
                # The original node is both weights and activation configurable
                self.activation_weights(activation_node, weights_node, virtual_node, origin_conf_nodes,
                                        virtual_cfg_idx, virtual_mp_cfg, virtual_sorted_nodes_names,
                                        origin_node_idx_to_cfg)
            else:
                # if weights_node.name in origin_conf_nodes:
                #     # It is possible that the original weights node is not configurable, in this case we don't need
                #     # to retrieve its bit-width config
                self.weights_only(weights_node, virtual_node, origin_conf_nodes, virtual_cfg_idx,
                                  origin_node_idx_to_cfg)
        else:
            assert isinstance(activation_node, VirtualSplitActivationNode)  # Sanity check
            if activation_node.name in virtual_sorted_nodes_names:
                self.activation_weights(activation_node, weights_node, virtual_node, origin_conf_nodes, virtual_cfg_idx,
                                        virtual_mp_cfg, virtual_sorted_nodes_names, origin_node_idx_to_cfg)
            else:
                # The original node is only weights configurable
                # self.weights_only(weights_node, virtual_node, origin_conf_nodes, virtual_cfg_idx, origin_node_idx_to_cfg)
                self.weights_only(weights_node.origin_node, virtual_node, origin_conf_nodes, virtual_cfg_idx, origin_node_idx_to_cfg)

    def get_weights_for_split_activation(self, activation_node, virtual_node, virtual_cfg_idx, virtual_sorted_nodes_names, virtual_mp_cfg, origin_conf_nodes, origin_node_idx_to_cfg):
        # This is an activation node that was split, means it has a weights node that should come before it,
        # and we need its configuration in order to reconstruct the original node's configuration.
        matching_weights_node = self.graph.get_prev_nodes(virtual_node)
        assert len(matching_weights_node) == 1
        weights_node = matching_weights_node[0]

        if isinstance(weights_node, VirtualActivationWeightsNode):
            if weights_node.original_weights_node.is_weights_quantization_enabled() and not \
                    weights_node.original_weights_node.is_all_weights_candidates_equal():
                assert weights_node.name in virtual_sorted_nodes_names  # Sanity check
                # The original node is both weights and activation configurable
                self.weights_activation(activation_node, weights_node, virtual_node, origin_conf_nodes, virtual_cfg_idx,
                                        virtual_mp_cfg, virtual_sorted_nodes_names, origin_node_idx_to_cfg)
            else:
                # The original node is only activation configurable
                self.activation_only(activation_node.origin_node, virtual_node, origin_conf_nodes, virtual_cfg_idx, origin_node_idx_to_cfg)
                # self.activation_only(activation_node, virtual_node, origin_conf_nodes, virtual_cfg_idx, origin_node_idx_to_cfg)
        else:
            # If the node's predecessor e multiple outgoing edges than it is possible that this weights
            # node is not composed with an activation, but otherwise this is something wrong and we need
            # to raise an exception
            predecessor = self.graph.get_prev_nodes(weights_node)
            assert len(predecessor) == 1  # Sanity check
            predecessor = predecessor[0]
            if len(self.graph.out_edges(predecessor)) > 1:
                # It's ok, need to find the node's configuration
                self.weights_activation(activation_node, weights_node, virtual_node, origin_conf_nodes, virtual_cfg_idx,
                                        virtual_mp_cfg, virtual_sorted_nodes_names, origin_node_idx_to_cfg)
            else:
                # Something wrong
                raise Exception()

    def reconstruct_config_from_virtual_graph(self, virtual_mp_cfg):
        origin_conf_nodes = self.original_graph.get_configurable_sorted_nodes_names()
        virtual_sorted_nodes_names = self.graph.get_configurable_sorted_nodes_names()

        origin_node_idx_to_cfg = {}
        for virtual_node_idx, n in enumerate(self.graph.get_configurable_sorted_nodes()):
            virtual_cfg_idx = virtual_mp_cfg[virtual_node_idx]

            if isinstance(n, VirtualActivationWeightsNode):
                weights_node = n.original_weights_node
                if isinstance(weights_node, VirtualSplitWeightsNode):
                    self.get_activation_for_split_weights(weights_node, n, virtual_cfg_idx, virtual_sorted_nodes_names, virtual_mp_cfg, origin_conf_nodes, origin_node_idx_to_cfg)
                else:
                    # if weights_node.name in origin_conf_nodes:
                    #     # It is possible that the original weights node is not configurable, in this case we don't need
                    #     # to retrieve its bit-width config
                    self.weights_only(weights_node, n, origin_conf_nodes, virtual_cfg_idx, origin_node_idx_to_cfg)

                activation_node = n.original_activation_node
                if isinstance(activation_node, VirtualSplitActivationNode):
                    self.get_weights_for_split_activation(activation_node, n, virtual_cfg_idx, virtual_sorted_nodes_names, virtual_mp_cfg, origin_conf_nodes, origin_node_idx_to_cfg)
                else:
                    if activation_node.name in origin_conf_nodes:
                        # It is possible that the original activation node is not configurable,
                        # in this case we don't need to retrieve its bit-width config
                        self.activation_only(activation_node, n, origin_conf_nodes, virtual_cfg_idx, origin_node_idx_to_cfg)
            elif isinstance(n, VirtualSplitWeightsNode):
                # If the node's predecessor e multiple outgoing edges than it is possible that this weights
                # node is not composed with an activation, but otherwise this is something wrong and we need
                # to raise an exception
                predecessor = self.graph.get_prev_nodes(n)
                assert len(predecessor) == 1  # Sanity check
                predecessor = predecessor[0]
                if len(self.graph.out_edges(predecessor)) > 1:
                    # It's ok, need to find the node's configuration
                    self.get_activation_for_split_weights(n, n, virtual_cfg_idx, virtual_sorted_nodes_names, virtual_mp_cfg, origin_conf_nodes, origin_node_idx_to_cfg)
                else:
                    # Something wrong
                    raise Exception()
            elif isinstance(n, VirtualSplitActivationNode):
                self.get_weights_for_split_activation(n, n, virtual_cfg_idx, virtual_sorted_nodes_names, virtual_mp_cfg, origin_conf_nodes, origin_node_idx_to_cfg)
            else:
                if n.name not in origin_conf_nodes:
                    # TODO: all nodes should be configurable in the original graph
                    raise Exception()
                origin_idx = origin_conf_nodes.index(n.name)
                origin_node_idx_to_cfg[origin_idx] = virtual_cfg_idx

        return [origin_node_idx_to_cfg[key] for key in sorted(origin_node_idx_to_cfg.keys())]


    # def reconstruct_config_from_virtual_graph(self, virtual_mp_cfg):
    #     origin_conf_nodes = self.original_graph.get_configurable_sorted_nodes_names()
    #     virtual_sorted_nodes_names = self.graph.get_configurable_sorted_nodes_names()
    #
    #     origin_node_idx_to_cfg = {}
    #     for virtual_node_idx, n in enumerate(self.graph.get_configurable_sorted_nodes()):
    #         virtual_cfg_idx = virtual_mp_cfg[virtual_node_idx]
    #
    #         if isinstance(n, VirtualActivationWeightsNode):
    #             # self.reconstruct_from_composed_node(n, origin_node_idx_to_cfg)
    #             weights_node = n.original_weights_node
    #             if isinstance(weights_node, VirtualSplitWeightsNode):
    #                 # This is a weights node that was split, means it has an activation node that should follow it,
    #                 # and we need its configuration in order to reconstruct the original node's configuration.
    #                 matching_activation_node = self.graph.get_next_nodes(n)
    #                 assert len(matching_activation_node) == 1
    #                 activation_node = matching_activation_node[0]
    #
    #                 if isinstance(activation_node, VirtualActivationWeightsNode):
    #                     # if activation_node.name in virtual_sorted_nodes_names:
    #                     if activation_node.original_activation_node.is_activation_quantization_enabled() and not \
    #                             activation_node.original_activation_node.is_all_activation_candidates_equal():
    #
    #                         assert activation_node.name in virtual_sorted_nodes_names  # Sanity check
    #
    #                         # The original node is both weights and activation configurable
    #                         activation_bitwidth = activation_node.candidates_quantization_cfg[virtual_mp_cfg[virtual_sorted_nodes_names.index(activation_node.name)]].activation_quantization_cfg.activation_n_bits
    #                         weights_bitwidth = n.candidates_quantization_cfg[virtual_cfg_idx].weights_quantization_cfg.weights_n_bits
    #                         origin_cfg_idx = [i for i, c in
    #                                           enumerate(weights_node.origin_node.candidates_quantization_cfg) if
    #                                           c.weights_quantization_cfg.weights_n_bits == weights_bitwidth and
    #                                           c.activation_quantization_cfg.activation_n_bits == activation_bitwidth]
    #
    #                         origin_idx = origin_conf_nodes.index(weights_node.origin_node.name)
    #                         origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx[0]
    #                     else:
    #                         # The original node is only weights configurable
    #                         weights_bitwidth = n.candidates_quantization_cfg[virtual_cfg_idx].weights_quantization_cfg.weights_n_bits
    #                         origin_cfg_idx = [i for i, c in
    #                                           enumerate(weights_node.origin_node.candidates_quantization_cfg) if
    #                                           c.weights_quantization_cfg.weights_n_bits == weights_bitwidth]
    #
    #                         origin_idx = origin_conf_nodes.index(weights_node.origin_node.name)
    #                         origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx[0]
    #                 else:
    #                     assert isinstance(activation_node, VirtualSplitActivationNode)  # Sanity check
    #                     if activation_node.name in virtual_sorted_nodes_names:
    #                         activation_bitwidth = activation_node.candidates_quantization_cfg[virtual_mp_cfg[virtual_sorted_nodes_names.index(activation_node.name)]].activation_quantization_cfg.activation_n_bits
    #                         weights_bitwidth = n.candidates_quantization_cfg[
    #                             virtual_cfg_idx].weights_quantization_cfg.weights_n_bits
    #                         origin_cfg_idx = [i for i, c in
    #                                           enumerate(weights_node.origin_node.candidates_quantization_cfg) if
    #                                           c.weights_quantization_cfg.weights_n_bits == weights_bitwidth and
    #                                           c.activation_quantization_cfg.activation_n_bits == activation_bitwidth]
    #
    #                         origin_idx = origin_conf_nodes.index(weights_node.origin_node.name)
    #                         origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx[0]
    #                     else:
    #                         # The original node is only weights configurable
    #                         weights_bitwidth = n.candidates_quantization_cfg[virtual_cfg_idx].weights_quantization_cfg.weights_n_bits
    #                         origin_cfg_idx = [i for i, c in
    #                                           enumerate(weights_node.origin_node.candidates_quantization_cfg) if
    #                                           c.weights_quantization_cfg.weights_n_bits == weights_bitwidth]
    #
    #                         origin_idx = origin_conf_nodes.index(weights_node.origin_node.name)
    #                         origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx[0]
    #
    #             else:
    #                 if weights_node.name in origin_conf_nodes:
    #                     # It is possible that the original weights node is not configurable, in this case we don't need
    #                     # to retrieve its bit-width config
    #                     weights_bitwidth = n.candidates_quantization_cfg[virtual_cfg_idx].weights_quantization_cfg.weights_n_bits
    #                     origin_cfg_idx = [i for i, c in
    #                                       enumerate(weights_node.candidates_quantization_cfg) if
    #                                       c.weights_quantization_cfg.weights_n_bits == weights_bitwidth]
    #
    #                     origin_idx = origin_conf_nodes.index(weights_node.name)
    #                     origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx[0]
    #
    #             activation_node = n.original_activation_node
    #             if isinstance(activation_node, VirtualSplitActivationNode):
    #                 # This is an activation node that was split, means it has a weights node that should come before it,
    #                 # and we need its configuration in order to reconstruct the original node's configuration.
    #                 matching_weights_node = self.graph.get_prev_nodes(n)
    #                 assert len(matching_weights_node) == 1
    #                 weights_node = matching_weights_node[0]
    #
    #                 if isinstance(weights_node, VirtualActivationWeightsNode):
    #                     # if weights_node.name in virtual_sorted_nodes_names:
    #                     if weights_node.original_weights_node.is_weights_quantization_enabled() and not \
    #                             weights_node.original_weights_node.is_all_weights_candidates_equal():
    #
    #                         assert weights_node.name in virtual_sorted_nodes_names  # Sanity check
    #
    #                         # The original node is both weights and activation configurable
    #                         weights_bitwidth = weights_node.candidates_quantization_cfg[virtual_mp_cfg[virtual_sorted_nodes_names.index(weights_node.name)]].weights_quantization_cfg.weights_n_bits
    #                         activation_bitwidth = n.candidates_quantization_cfg[virtual_cfg_idx].activation_quantization_cfg.activation_n_bits
    #                         origin_cfg_idx = [i for i, c in enumerate(activation_node.origin_node.candidates_quantization_cfg) if
    #                                           c.weights_quantization_cfg.weights_n_bits == weights_bitwidth and
    #                                           c.activation_quantization_cfg.activation_n_bits == activation_bitwidth]
    #
    #                         if activation_node.origin_node.name not in origin_conf_nodes:
    #                             # TODO: all nodes should be configurable in the original graph
    #                             raise Exception()
    #                         origin_idx = origin_conf_nodes.index(activation_node.origin_node.name)
    #                         origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx[0]
    #                     else:
    #                         # The original node is only activation configurable
    #                         activation_bitwidth = n.candidates_quantization_cfg[virtual_cfg_idx].activation_quantization_cfg.activation_n_bits
    #                         origin_cfg_idx = [i for i, c in
    #                                           enumerate(activation_node.candidates_quantization_cfg) if
    #                                           c.activation_quantization_cfg.activation_n_bits == activation_bitwidth]
    #
    #                         if activation_node.origin_node.name not in origin_conf_nodes:
    #                             # TODO: all nodes should be configurable in the original graph
    #                             raise Exception()
    #                         origin_idx = origin_conf_nodes.index(activation_node.origin_node.name)
    #                         origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx[0]
    #                 else:
    #                     # TODO: not supposed to happen since a linear operation always follows an activation - raise exception
    #                     #  unless maybe its predecessor has multiple outputs
    #                     raise Exception()
    #             else:
    #                 if activation_node.name in origin_conf_nodes:
    #                     # It is possible that the original activation node is not configurable,
    #                     # in this case we don't need to retrieve its bit-width config
    #                     activation_bitwidth = n.candidates_quantization_cfg[virtual_cfg_idx].activation_quantization_cfg.activation_n_bits
    #                     origin_cfg_idx = [i for i, c in
    #                                       enumerate(activation_node.candidates_quantization_cfg) if
    #                                       c.activation_quantization_cfg.activation_n_bits == activation_bitwidth]
    #
    #                     origin_idx = origin_conf_nodes.index(activation_node.name)
    #                     origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx[0]
    #
    #         elif isinstance(n, VirtualSplitWeightsNode):
    #             # TODO: not supposed to happen since a linear operation always follows an activation - raise exception
    #             #  unless maybe its predecessor has multiple outputs
    #             ancestor = self.graph.get_prev_nodes(n)
    #             assert len(ancestor) == 1
    #             ancestor = ancestor[0]
    #             if len(self.graph.out_edges(ancestor)) > 1:
    #                 # This node is not composed with an activation node because its ancestor has multiple outgoing edges
    #                 # TODO: need to take care of finding its matching activation
    #                 pass
    #             else:
    #                 raise Exception()
    #         elif isinstance(n, VirtualSplitActivationNode):
    #             # This is an activation node that was split, means it has a weights node that should come before it,
    #             # and we might need its configuration in order to reconstruct the original node's configuration.
    #             matching_weights_node = self.graph.get_prev_nodes(n)
    #             assert len(matching_weights_node) == 1
    #             weights_node = matching_weights_node[0]
    #
    #             if isinstance(weights_node, VirtualActivationWeightsNode):
    #                 # if weights_node.name in virtual_sorted_nodes_names:
    #                 if weights_node.original_weights_node.is_weights_quantization_enabled() and not \
    #                         weights_node.original_weights_node.is_all_weights_candidates_equal():
    #
    #                     assert weights_node.name in virtual_sorted_nodes_names  # Sanity check
    #
    #                     # The original node is both weights and activation configurable
    #                     weights_bitwidth = weights_node.candidates_quantization_cfg[virtual_mp_cfg[virtual_sorted_nodes_names.index(weights_node.name)]].weights_quantization_cfg.weights_n_bits
    #                     activation_bitwidth = n.candidates_quantization_cfg[virtual_cfg_idx].activation_quantization_cfg.activation_n_bits
    #                     origin_cfg_idx = [i for i, c in enumerate(n.origin_node.candidates_quantization_cfg) if
    #                                       c.weights_quantization_cfg.weights_n_bits == weights_bitwidth and
    #                                       c.activation_quantization_cfg.activation_n_bits == activation_bitwidth]
    #
    #                     if n.origin_node.name not in origin_conf_nodes:
    #                         # TODO: all nodes should be configurable in the original graph
    #                         raise Exception()
    #                     origin_idx = origin_conf_nodes.index(n.origin_node.name)
    #                     origin_node_idx_to_cfg[origin_idx] = origin_cfg_idx[0]
    #                 else:
    #                     # The original node is only activation configurable
    #                     activation_bitwidth = n.candidates_quantization_cfg[virtual_cfg_idx].activation_quantization_cfg.activation_n_bits
    #                     origin_cfg_idx = [i for i, c in
    #                                       enumerate(n.candidates_quantization_cfg) if
    #                                       c.activation_quantization_cfg.activation_n_bits == activation_bitwidth]
    #
    #                     if n.origin_node.name not in origin_conf_nodes:
    #                         # TODO: all nodes should be configurable in the original graph
    #                         raise Exception()
    #                     origin_idx = origin_conf_nodes.index(n.origin_node.name)
    #                     origin_node_idx_to_cfg[origin_idx] = virtual_cfg_idx
    #
    #             else:
    #                 # TODO: not supposed to happen since a linear operation always follows an activation - raise exception
    #                 #  unless maybe its predecessor has multiple outputs
    #                 raise Exception()
    #         else:
    #             if n.name not in origin_conf_nodes:
    #                 # TODO: all nodes should be configurable in the original graph
    #                 raise Exception()
    #             origin_idx = origin_conf_nodes.index(n.name)
    #             origin_node_idx_to_cfg[origin_idx] = virtual_cfg_idx
    #
    #     return [origin_node_idx_to_cfg[key] for key in sorted(origin_node_idx_to_cfg.keys())]

