# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from typing import Tuple

import tensorflow as tf
import copy
from tensorflow.keras.layers import Dense, DepthwiseConv2D, Conv2D, Conv2DTranspose
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.constants import DEFAULT_CANDIDATE_BITWIDTH
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, NodeFrameworkAttrMatcher, \
    EdgeMatcher
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.keras.constants import LINEAR, ACTIVATION


class VirtualActivationWeightsComposition(common.BaseSubstitution):
    def __init__(self):
        """
        Matches: (DepthwiseConv2D, Conv2D, Dense, Conv2DTranspose, SeparableConv2D)[activation != identity]
        """

        # We assume that the graph that this substitution is running on had all its kernel node
        # have been decomposed from their activations
        weights_node = NodeOperationMatcher(DepthwiseConv2D) | \
                       NodeOperationMatcher(Conv2D) | \
                       NodeOperationMatcher(Dense) | \
                       NodeOperationMatcher(Conv2DTranspose)

        act_node = weights_node.logic_not()

        super().__init__(matcher_instance=EdgeMatcher(act_node, weights_node))

    def substitute(self,
                   graph: Graph,
                   edge_nodes: Tuple[BaseNode, BaseNode]) -> Graph:
        """
        Decompose the activation function in a linear node to a new activation layer.
        Set activation function in the linear node to 'linear' (y=x).

        Args:
            graph: Graph we apply the substitution on.
            op2d_node: Node to extract its activation function.

        Returns:
            Graph after applying the substitution.
        """

        act_node = edge_nodes[0]
        weights_node = edge_nodes[1]

        # Virtual composed activation-weights node
        v_node = copy.deepcopy(weights_node)
        v_node.name = f"virtual_{act_node.name}_{weights_node.name}"

        v_candidates = []
        for c_a in act_node.candidates_quantization_cfg:
            for c_w in weights_node.candidates_quantization_cfg:
                composed_candidate = CandidateNodeQuantizationConfig(activation_quantization_cfg=c_a,
                                                                     weights_quantization_cfg=c_w)
                v_candidates.append(composed_candidate)

        v_node.candidates_quantization_cfg = v_candidates

        # TODO: how do we preserve the original activation operation **before** the convolution when we compose it to a single node?

        # Update graph
        # TODO: complete implementing graph update after building the node
        # graph.add_node(weights_node)
        # graph.add_node(activation_node)
        # graph.reconnect_in_edges(current_node=node, new_node=weights_node)
        # graph.reconnect_out_edges(current_node=node, new_node=activation_node)
        # graph.replace_output_node(current_node=node, new_node=activation_node)
        # graph.add_edge(weights_node, activation_node)
        # graph.remove_node(node)

        return graph

