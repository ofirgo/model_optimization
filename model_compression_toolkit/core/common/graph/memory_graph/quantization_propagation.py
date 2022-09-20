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
from typing import Dict, List, Callable

from model_compression_toolkit.core.common import Graph, BaseNode, Logger
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig


def forward_propagation(graph: Graph,
                        forward_propagatable: Callable) -> Dict[BaseNode, List[CandidateNodeQuantizationConfig]]:
    node_to_candidates = {}
    topo_sorted_nodes = graph.get_topo_sorted_nodes()
    fused_nodes_flat = list([n for fusing in graph.fused_nodes for n in fusing])
    fused_nodes_flat = set() if len(fused_nodes_flat) == 0 else list(set(fused_nodes_flat))

    for i, n in enumerate(topo_sorted_nodes[1:]):
        # TODO: should we look at each node's incoming_edges instead of working with the topo sort,
        #  to "catch" nodes with multiple inputs? (which inputs' candidates we propagate then?)
        #  or at least ignore nodes with multiple inputs?

        if n.get_has_activation() and not n.has_activation_quantization_enabled_candidate():
            if forward_propagatable(n):
                prev_node = topo_sorted_nodes[i-1]
                prop_candidates = node_to_candidates.get(prev_node, prev_node.candidates_quantization_cfg)
                node_to_candidates[n] = prop_candidates
            elif n in fused_nodes_flat:
                # Node matches a fusing pattern, this is taken care of during backward propagation
                continue
            else:
                Logger.critical("Found a node without activation quantization candidates that is not matching "
                                "for neither forward nor backward quantization propagation.")
    return node_to_candidates


def backward_propagation(graph: Graph, forward_propagatable: Callable) -> Dict[BaseNode, List[CandidateNodeQuantizationConfig]]:
    node_to_candidates = {}
    topo_sorted_nodes = graph.get_topo_sorted_nodes()
    for i, n in enumerate(topo_sorted_nodes[1:]):
        # TODO: should we look at each node's outgoing_edges instead of working with the topo sort,
        #  to "catch" nodes with multiple outputs? (which outputs candidates we propagate then?)
        #  or at least ignore nodes with multiple inputs?

        if n.get_has_activation() and not n.has_activation_quantization_enabled_candidate():
            node_fusion = [fusion for fusion in graph.fused_nodes if n in fusion]
            if len(node_fusion) > 0:
                if len(node_fusion) > 1:
                    Logger.critical("Node can't appear in more than one fusion.")
                node_fusion = node_fusion[0]
                last_node_in_fusion = node_fusion[:-1]
                prop_candidates = node_to_candidates.get(last_node_in_fusion, last_node_in_fusion.candidates_quantization_cfg)
                node_to_candidates[n] = prop_candidates
            elif forward_propagatable(n):
                # Node is forward propagatable, this is taken care of during forward propagation
                continue
            else:
                Logger.critical("Found a node without activation quantization candidates that is not matching "
                                "for neither forward nor backward quantization propagation.")
    return node_to_candidates