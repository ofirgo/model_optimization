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

from tensorflow.keras.layers import Dense, DepthwiseConv2D, Conv2D, Conv2DTranspose
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, EdgeMatcher
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualActivationWeightsNode


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

        # if isinstance(weights_node, VirtualActivationWeightsNode):
        #     return graph

        if len(graph.out_edges(act_node)) > 1:
            Logger.warning(f"Node {act_node.name} has multiple outgoing edges, which is not supported with "
                           f"mixed-precision bit-operations KPI, thus, edge {act_node.name} --> {weights_node.name} "
                           f"would not be counted in the bit-operations calculations.")
            return graph

        topo_sorted_nodes_names = [n.name for n in graph.get_topo_sorted_nodes()]
        sorted_conf_nodes_names = graph.get_configurable_sorted_nodes_names()

        original_act_node_idx = topo_sorted_nodes_names.index(act_node.name)
        original_weights_node_idx = topo_sorted_nodes_names.index(weights_node.name)

        conf_act_node_idx = sorted_conf_nodes_names.index(act_node.name) \
                           if act_node.name in sorted_conf_nodes_names else None,
        conf_weights_node_idx = sorted_conf_nodes_names.index(weights_node.name) \
                               if weights_node.name in sorted_conf_nodes_names else None,

        # Virtual composed activation-weights node
        # we pass a dummy initialization dict to initialize the super BaseNode class,
        # the actual arguments values are irrelevant because they are being overridden or not used
        v_node = VirtualActivationWeightsNode(act_node,
                                              weights_node,
                                              # original_act_node_idx,
                                              # original_weights_node_idx,
                                              # conf_act_node_idx,
                                              # conf_weights_node_idx,
                                              **weights_node.__dict__)

        # Update graph
        graph.add_node(v_node)
        graph.reconnect_in_edges(current_node=act_node, new_node=v_node)
        graph.reconnect_out_edges(current_node=weights_node, new_node=v_node)
        graph.replace_input_node(current_node=act_node, new_node=v_node)
        graph.replace_output_node(current_node=weights_node, new_node=v_node)
        graph.remove_edge(act_node, weights_node)
        graph.remove_node(weights_node)
        graph.remove_node(act_node)

        return graph

