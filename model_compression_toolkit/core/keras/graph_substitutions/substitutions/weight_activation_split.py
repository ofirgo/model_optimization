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

import tensorflow as tf
import copy
from tensorflow.keras.layers import Dense, DepthwiseConv2D, Conv2D, Conv2DTranspose
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.constants import DEFAULT_CANDIDATE_BITWIDTH
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.keras.constants import LINEAR, ACTIVATION


class WeightsActivationSplit(common.BaseSubstitution):
    def __init__(self):
        """
        Matches: (DepthwiseConv2D, Conv2D, Dense, Conv2DTranspose, SeparableConv2D)[activation != identity]
        """
        op2d_node = NodeOperationMatcher(DepthwiseConv2D) | \
                    NodeOperationMatcher(Conv2D) | \
                    NodeOperationMatcher(Dense) | \
                    NodeOperationMatcher(Conv2DTranspose)

        op2d_node = op2d_node
        super().__init__(matcher_instance=op2d_node)

    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:
        """
        Decompose the activation function in a linear node to a new activation layer.
        Set activation function in the linear node to 'linear' (y=x).

        Args:
            graph: Graph we apply the substitution on.
            op2d_node: Node to extract its activation function.

        Returns:
            Graph after applying the substitution.
        """

        if node.is_all_weights_candidates_equal() or node.is_all_activation_candidates_equal():
            # If the node is has only one of weights or activation to configure (or none) then it shouldn't be split.
            return graph

        # TODO: verify that it is also composite

        # If we arrived here, then we have a node with configurable weights and activation
        # Virtual weights node
        weights_node = copy.deepcopy(node)
        weights_node.name = node.name + '_v_weights'  # 'v' is for 'virtual'
        weights_node.candidates_quantization_cfg = node.get_unique_weights_candidates()

        for c in weights_node.candidates_quantization_cfg:
            c.activation_quantization_cfg.enable_activation_quantization = False
            c.activation_quantization_cfg.activation_n_bits = DEFAULT_CANDIDATE_BITWIDTH

        # Virtual activation node
        # This is an identity node that passes the convolution output and quantize it if needed.
        # TODO: should we do something different if the node is 'reused'?
        activation_node = BaseNode(name=node.name + '_v_activation',  # 'v' is for 'virtual'
                                   framework_attr={ACTIVATION: LINEAR},
                                   input_shape=node.output_shape,  # the kernel output in the activation input
                                   output_shape=node.output_shape,
                                   weights=None,
                                   layer_class=tf.keras.layers.Activation,
                                   reuse=node.reuse,
                                   reuse_group=node.reuse_group,
                                   quantization_attr=node.quantization_attr)

        activation_node.prior_info = node.prior_info
        activation_node.candidates_quantization_cfg = node.get_unique_activation_candidates()

        for c in activation_node.candidates_quantization_cfg:
            c.activation_quantization_cfg.enable_weights_quantization = False
            c.activation_quantization_cfg.weights_n_bits = DEFAULT_CANDIDATE_BITWIDTH

        # Update graph
        graph.add_node(weights_node)
        graph.add_node(activation_node)
        graph.reconnect_in_edges(current_node=node, new_node=weights_node)
        graph.reconnect_out_edges(current_node=node, new_node=activation_node)
        graph.replace_output_node(current_node=node, new_node=activation_node)
        graph.add_edge(weights_node, activation_node)
        graph.remove_node(node)

        return graph

