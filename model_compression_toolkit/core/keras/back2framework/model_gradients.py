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
import numpy as np
import tensorflow as tf

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers import Layer
else:
    from keras.engine.base_layer import Layer

from typing import Any, Dict, List
from tensorflow.python.util.object_identity import Reference as TFReference
from model_compression_toolkit.core.common.constants import EPS
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.keras.back2framework.instance_builder import OperationHandler
from model_compression_toolkit.core.common.logger import Logger


@tf.RegisterGradient("ResizeBilinearGrad")
def _ResizeBilinearGrad_grad(op, grad):
    up = tf.image.resize(grad, tf.shape(op.inputs[0])[1:-1])
    return up, None


def build_input_tensors_list(node: BaseNode,
                             graph: Graph,
                             node_to_output_tensors_dict: Dict[BaseNode, List[TFReference]]) -> List[List[TFReference]]:
    """
    Given a node, build a list of input tensors the node gets. The list is built
    based on the node's incoming edges and previous nodes' output tensors.

    Args:
        node: Node to build its input tensors list.
        graph: Graph the node is in.
        node_to_output_tensors_dict: A dictionary from a node to its output tensors.

    Returns:
        A list of the node's input tensors.
    """

    input_tensors = []
    # Go over a sorted list of the node's incoming edges, and for each source node get its output tensors.
    # Append them in a result list.
    for ie in graph.incoming_edges(node, sort_by_attr=EDGE_SINK_INDEX):
        _input_tensors = [node_to_output_tensors_dict[ie.source_node][ie.source_index]]
        input_tensors.append(_input_tensors)
    return input_tensors


def run_operation(n: BaseNode,
                  input_tensors: List[List[TFReference]],
                  op_func: Layer,
                  input_nodes_to_input_tensors: Dict[BaseNode, Any]) -> List[TFReference]:
    """
    Applying the layer (op_func) to the input tensors (input_tensors).

    Args:
        n: The corresponding node of the layer it runs.
        input_tensors: List of references to Keras tensors that are the layer's inputs.
        op_func: Layer to apply to the input tensors.
        input_nodes_to_input_tensors: A dictionary from a node to its input tensors.

    Returns:
        A list of references to Keras tensors. The layer's output tensors after applying the
        layer to the input tensors.
    """

    if len(input_tensors) == 0:  # Placeholder handling
        out_tensors_of_n = input_nodes_to_input_tensors[n]
    else:
        input_tensors = [tensor for tensor_list in input_tensors for tensor in tensor_list]  # flat list of lists
        # Build a functional node using its args
        if isinstance(n, FunctionalNode):
            if n.inputs_as_list:  # If the first argument should be a list of tensors:
                out_tensors_of_n = op_func(input_tensors, *n.op_call_args, **n.op_call_kwargs)
            else:  # If the input tensors should not be a list but iterated:
                out_tensors_of_n = op_func(*input_tensors, *n.op_call_args, **n.op_call_kwargs)
        else:
            # If operator expects a single input tensor, it cannot be a list as it should have a dtype field.
            if len(input_tensors) == 1:
                input_tensors = input_tensors[0]
            out_tensors_of_n = op_func(input_tensors)

    return out_tensors_of_n


def keras_iterative_approx_hessian_trace(graph_float: common.Graph,
                                         model_input_tensors: Dict[BaseNode, np.ndarray],
                                         interest_points: List[BaseNode],
                                         output_list: List[BaseNode],
                                         all_outputs_indices: List[int],
                                         alpha: float = 0.3,
                                         num_iter: int = 50) -> List[float]:
    """
    Computes the gradients of a Keras model's outputs with respect to the feature maps of the set of given
    interest points. It then uses the gradients to compute the hessian trace for each interest point and normalized the
    values, to be used as weights for weighted average in mixed-precision distance metric computation.

    Args:
        graph_float: Graph to build its corresponding Keras model.
        model_input_tensors: A mapping between model input nodes to an input batch.
        interest_points: List of nodes which we want to get their feature map as output, to calculate distance metric.
        output_list: List of nodes that considered as model's output for the purpose of gradients computation.
        all_outputs_indices: Indices of the model outputs and outputs replacements (if exists),
            in a topological sorted interest points list.
        alpha: A tuning parameter to allow calibration between the contribution of the output feature maps returned
            weights and the other feature maps weights (since the gradient of the output layers does not provide a
            compatible weight for the distance metric computation).

    Returns: A list of normalized gradients to be considered as the relevancy that each interest
    point's output has on the model's output.
    """
    if not all([images.shape[0] == 1 for node, images in model_input_tensors.items()]):
        Logger.critical("Iterative hessian trace computation is only supported on a single image sample")
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gg:
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
            output_loss, interest_points_tensors, output_features_sum = _model_outputs_computation(graph_float,
                                                                                                   model_input_tensors,
                                                                                                   interest_points,
                                                                                                   output_list,
                                                                                                   gradient_tapes_list=[
                                                                                                       gg, g],
                                                                                                   output_loss_fn=lambda output:
                                                                                                   0.5 * tf.pow(tf.reduce_sum(output), 2))  # Single image in batch
        # End of inner GradientTape g

        hessian_trace_approx = []
        for ipt in interest_points_tensors:
            r_grad_ipt = g.gradient(output_loss, ipt, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            r_grad_ipt = tf.reshape(r_grad_ipt, shape=[ipt.shape[0], -1])

            trace_vhv = []
            for j in range(num_iter):
                v = tf.random.uniform(shape=r_grad_ipt.shape, minval=0, maxval=2, dtype=tf.int32)
                with gg.stop_recording():
                    mask = tf.where(v == 0)
                v = tf.tensor_scatter_nd_update(v, mask, tf.fill(mask.shape[0], -1))
                v = tf.dtypes.cast(v, tf.float32)

                # Compute inner product with gradient
                g_v = tf.matmul(tf.expand_dims(r_grad_ipt, axis=1), tf.expand_dims(v, axis=2))
                g_v_loss = tf.reduce_mean(g_v)

                with gg.stop_recording():
                    H_v = gg.gradient(g_v_loss, ipt, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                    H_v = tf.reshape(H_v, shape=[H_v.shape[0], -1])
                    trace_vhv.append(tf.reduce_mean(tf.matmul(tf.expand_dims(v, axis=1), tf.expand_dims(H_v, axis=2))))
            # hessian_trace_approx.append(tf.reduce_mean(trace_vhv))
            with gg.stop_recording():
                output_features_sum = tf.abs(output_features_sum)
                normalized_grad = r_grad_ipt / (output_features_sum + EPS)
                squared_grad_trace = tf.reduce_sum(tf.pow(normalized_grad, 2))
            hessian_trace_approx.append(tf.abs(tf.reduce_mean(trace_vhv) - squared_grad_trace) / (output_features_sum + EPS))

    return _normalize_weights(hessian_trace_approx, all_outputs_indices, alpha)


def keras_approx_hessian_trace(graph_float: common.Graph,
                               model_input_tensors: Dict[BaseNode, np.ndarray],
                               interest_points: List[BaseNode],
                               output_list: List[BaseNode],
                               all_outputs_indices: List[int],
                               alpha: float = 0.3) -> List[float]:

    with tf.GradientTape(persistent=True) as g:
        output_loss, interest_points_tensors, _ = _model_outputs_computation(graph_float,
                                                                             model_input_tensors,
                                                                             interest_points, output_list,
                                                                             gradient_tapes_list=[g],
                                                                             output_loss_fn=lambda output:
                                                                             tf.reduce_mean(0.5 * tf.reduce_sum(
                                                                                 tf.pow(output, 2), axis=-1)))

    hessian_scores = []
    for ipt in interest_points_tensors:
        grad_ipt = g.gradient(output_loss, ipt, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        r_grad_ipt = tf.reshape(grad_ipt, shape=[grad_ipt.shape[0], -1])
        hessian_trace_approx = tf.reduce_mean(tf.reduce_sum(tf.pow(r_grad_ipt, 2.0), axis=-1))
        hessian_scores.append(hessian_trace_approx)

    return _normalize_weights(hessian_scores, all_outputs_indices, alpha)


def _model_outputs_computation(graph_float,
                               model_input_tensors,
                               interest_points,
                               output_list,
                               gradient_tapes_list,
                               output_loss_fn):

    node_to_output_tensors_dict = dict()

    # Build an OperationHandler to handle conversions from graph nodes to Keras operators.
    oh = OperationHandler(graph_float)

    input_nodes_to_input_tensors = {inode: tf.convert_to_tensor(model_input_tensors[inode]) for
                                    inode in graph_float.get_inputs()}  # Cast numpy array to tf.Tensor

    interest_points_tensors = []
    output_tensors = []

    for n in oh.node_sort:
        # Build a dictionary from node to its output tensors, by applying the layers sequentially.
        op_func = oh.get_node_op_function(n)  # Get node operation function

        input_tensors = build_input_tensors_list(n,
                                                 graph_float,
                                                 node_to_output_tensors_dict)  # Fetch Node inputs
        out_tensors_of_n = run_operation(n,  # Run node operation and fetch outputs
                                         input_tensors,
                                         op_func,
                                         input_nodes_to_input_tensors)

        # Gradients can be computed only on float32 tensors
        out_tensors_of_n = tf.dtypes.cast(out_tensors_of_n, tf.float32)
        if n in interest_points:
            # Recording the relevant feature maps onto the gradient tape
            for g in gradient_tapes_list:
                g.watch(out_tensors_of_n)
            interest_points_tensors.append(out_tensors_of_n)
        if n in output_list:
            output_tensors.append(out_tensors_of_n)

        if isinstance(out_tensors_of_n, list):
            node_to_output_tensors_dict.update({n: out_tensors_of_n})
        else:
            node_to_output_tensors_dict.update({n: [out_tensors_of_n]})

    # Get a reduced loss value for derivatives computation
    output_loss = 0
    for output in output_tensors:
        output = tf.reshape(output, shape=(output.shape[0], -1))
        output_loss += output_loss_fn(output)

    return output_loss, interest_points_tensors, tf.add_n([tf.reduce_sum(ot) for ot in output_tensors])


def _normalize_weights(hessian_scores,
                       all_outputs_indices,
                       alpha):
    # Output layers or layers that come after the model's considered output layers,
    # are assigned with a constant normalized value,
    # according to the given alpha variable and the number of such layers.
    # Other layers returned weights are normalized by dividing the hessian value by the sum of all other values.
    sum_without_outputs = sum([hessian_scores[i] for i in range(len(hessian_scores)) if i not in all_outputs_indices])
    normalized_grads_weights = [_get_normalized_weight(grad, i, sum_without_outputs, all_outputs_indices, alpha)
                                for i, grad in enumerate(hessian_scores)]

    return normalized_grads_weights


def _get_normalized_weight(grad: float,
                           i: int,
                           sum_without_outputs: float,
                           all_outputs_indices: List[int],
                           alpha: float) -> float:
    """
    Normalizes the node's gradient value. If it is an output or output replacement node than the normalized value is
    a constant, otherwise, it is normalized by dividing with the sum of all gradient values.

    Args:
        grad: The gradient value.
        i: The index of the node in the sorted interest points list.
        sum_without_outputs: The sum of all gradients of nodes that are not considered outputs.
        all_outputs_indices: A list of indices of all nodes that consider outputs.
        alpha: A multiplication factor.

    Returns: A normalized gradient value.

    """

    if i in all_outputs_indices:
        return alpha / len(all_outputs_indices)
    else:
        return ((1 - alpha) * grad / (sum_without_outputs + EPS)).numpy()
