from typing import Dict, Any, Tuple, List

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.core.common.constants import VIRTUAL_ACTIVATION_WEIGHTS_NODE_PREFIX, \
    VIRTUAL_WEIGHTS_SUFFIX, DEFAULT_CANDIDATE_BITWIDTH, VIRTUAL_ACTIVATION_SUFFIX
from model_compression_toolkit.core.common.graph.base_node import BaseNode
import numpy as np

from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.keras.constants import ACTIVATION, LINEAR


class VirtualSplitNode(BaseNode):
    def __init__(self, origin_node):
        super().__init__(origin_node.name,
                         origin_node.framework_attr,
                         origin_node.input_shape,
                         origin_node.output_shape,
                         origin_node.weights,
                         origin_node.layer_class,
                         origin_node.reuse,
                         origin_node.reuse_group,
                         origin_node.quantization_attr,
                         origin_node.has_activation)

        self.origin_node = origin_node


class VirtualSplitWeightsNode(VirtualSplitNode):
    def __init__(self, origin_node):
        super().__init__(origin_node)

        self.name = origin_node.name + VIRTUAL_WEIGHTS_SUFFIX

        self.candidates_quantization_cfg = origin_node.get_unique_weights_candidates()
        for c in self.candidates_quantization_cfg:
            c.activation_quantization_cfg.enable_activation_quantization = False
            c.activation_quantization_cfg.activation_n_bits = DEFAULT_CANDIDATE_BITWIDTH


class VirtualSplitActivationNode(VirtualSplitNode):
    def __init__(self, origin_node, activation_class):
        super().__init__(origin_node)

        self.name = origin_node.name + VIRTUAL_ACTIVATION_SUFFIX
        self.framework_attr = {ACTIVATION: LINEAR}
        self.prior_info = origin_node.prior_info
        self.input_shape = origin_node.output_shape  # the kernel output is the activation input
        self.weights = {}
        self.layer_class = activation_class

        self.candidates_quantization_cfg = origin_node.get_unique_activation_candidates()
        for c in self.candidates_quantization_cfg:
            c.activation_quantization_cfg.enable_weights_quantization = False
            c.activation_quantization_cfg.weights_n_bits = DEFAULT_CANDIDATE_BITWIDTH


class VirtualActivationWeightsNode(BaseNode):
    """
    Node that represents function ops with arguments to pass when building back the model.
    """

    def __init__(self,
                 act_node: BaseNode,
                 weights_node: BaseNode,
                 # original_act_node_idx: int,
                 # original_weights_node_idx: int,
                 # conf_act_node_idx: int,
                 # conf_weights_node_idx: int,
                 name: str,
                 framework_attr: Dict[str, Any],
                 input_shape: Tuple[Any],
                 output_shape: Tuple[Any],
                 weights: Dict[str, np.ndarray],
                 layer_class: type,
                 reuse: bool = False,
                 reuse_group: str = None,
                 quantization_attr: Dict[str, Any] = None,
                 has_activation: bool = True,
                 **kwargs):
        """
        Init a FunctionalNode object.

        Args:
            name: Node's name
            framework_attr: Framework attributes the layer had which the node holds.
            input_shape: Input tensor shape of the node.
            output_shape: Input tensor shape of the node.
            weights: Dictionary from a variable name to the weights with that name in the layer the node represents.
            layer_class: Class path of the layer this node represents.
            reuse: Whether this node was duplicated and represents a reused layer.
            reuse_group: Name of group of nodes from the same reused layer.
            quantization_attr: Attributes the node holds regarding how it should be quantized.
            has_activation: Whether the node has activations that we might want to quantize.

        """

        super().__init__(name,
                         framework_attr,
                         input_shape,
                         output_shape,
                         weights,
                         layer_class,
                         reuse,
                         reuse_group,
                         quantization_attr,
                         has_activation)

        self.name = f"{VIRTUAL_ACTIVATION_WEIGHTS_NODE_PREFIX}_{act_node.name}_{weights_node.name}"

        self.original_activation_node = act_node
        self.original_weights_node = weights_node

        v_candidates = []
        for c_a in act_node.candidates_quantization_cfg:
            for c_w in weights_node.candidates_quantization_cfg:
                composed_candidate = CandidateNodeQuantizationConfig(activation_quantization_cfg=c_a.activation_quantization_cfg,
                                                                     weights_quantization_cfg=c_w.weights_quantization_cfg)
                v_candidates.append(composed_candidate)

        # sorting the candidates by weights number of bits first and then by activation number of bits (reversed order)
        v_candidates.sort(key=lambda c: (c.weights_quantization_cfg.weights_n_bits,
                                         c.activation_quantization_cfg.activation_n_bits), reverse=True)

        self.candidates_quantization_cfg = v_candidates



        # # To allow matching the composed node to its building blocks from the original graph.
        # self.original_act_node_idx = original_act_node_idx
        # self.original_weights_node_idx = original_weights_node_idx
        #
        # # To allow retrieving the nodes' from the original configurable nodes list (if they were configurable)
        # self.conf_act_node_idx = conf_act_node_idx
        # self.conf_weights_node_idx = conf_weights_node_idx

        # Overriding the node's class to prevent using the original node's operation mistakenly.
        # The node should not being used for inference, but just for reconstructing the original graph, therefore,
        # it's not necessary t0 maintain its layer operation.
        # self.layer_class = None

        # TODO: disable activation quantization of the OUTPUT of the composed node (but keep quantization of the
        #   intermediate activation if it was originally enabled)

    def is_original_activation_node_configurable(self):
        return self.conf_act_node_idx is not None

    def is_original_weights_node_configurable(self):
        return self.conf_weights_node_idx is not None

    def conf_activation_node_index(self):
        return self.conf_act_node_idx

    def conf_weights_node_index(self):
        return self.conf_weights_node_idx

    def original_activation_node_index(self):
        return self.original_act_node_idx

    def original_weights_node_index(self):
        return self.original_weights_node_idx

    def get_bops_count(self, fw_impl: Any, fw_info: FrameworkInfo, candidate_idx: int):
        node_mac = fw_impl.get_node_mac_operations(self.original_weights_node, fw_info)
        node_bops = self.candidates_quantization_cfg[candidate_idx].weights_quantization_cfg.weights_n_bits * \
                    self.candidates_quantization_cfg[candidate_idx].activation_quantization_cfg.activation_n_bits * \
                    node_mac
        return node_bops
