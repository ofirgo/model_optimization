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
from operator import itemgetter

import numpy as np
from pulp import *
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable

from model_compression_toolkit.common import Logger
from model_compression_toolkit.common.mixed_precision.kpi import KPI


def mp_dynamic_programming_mckp_search(layer_to_bitwidth_mapping: Dict[int, List[int]],
                                       compute_metric_fn: Callable,
                                       compute_kpi_fn: Callable,
                                       target_kpi: KPI = None) -> List[int]:
    """
    Searching and returning a mixed-precision configuration using an ILP optimization solution.
    It first builds a mapping from each layer's index (in the model) to a dictionary that maps the
    bitwidth index to the observed sensitivity of the model when using that bitwidth for that layer.
    Then, it creates a mapping from each node's index (in the graph) to a dictionary
    that maps the bitwidth index to the contribution of configuring this node with this
    bitwidth to the minimal possible KPI of the model.
    Then, and using these mappings, it builds an LP problem and finds an optimal solution.
    If a solution could not be found, exception is thrown.

    Args:
        layer_to_bitwidth_mapping: Search space (mapping from each node's index to its possible bitwidth
        indices).
        compute_metric_fn: Function to compute a metric for a mixed-precision model configuration.
        compute_kpi_fn: Function to compute the KPI of the model for some mixed-precision configuration.
        target_kpi: KPI to constrain our LP problem with some resources limitations (like model' weights memory
        consumption).

    Returns:
        The mixed-precision configuration (list of indices. Each indicates the bitwidth index of a node).

    """
    if np.isinf(target_kpi.weights_memory):
        return [min(bitwidth_candidates) for _, bitwidth_candidates in layer_to_bitwidth_mapping.items()]

    # Build a mapping from each layer's index (in the model) to a dictionary that maps the
    # bitwidth index to the observed sensitivity of the model when using that bitwidth for that layer.
    layer_to_metrics_mapping = _build_layer_to_metrics_mapping(layer_to_bitwidth_mapping,
                                                               compute_metric_fn)

    # Init variables to find their values when solving the lp problem.
    # layer_to_indicator_vars_mapping, layer_to_objective_vars_mapping = _init_problem_vars(layer_to_metrics_mapping)

    # Build a mapping from each node's index (in the graph) to a dictionary
    # that maps the bitwidth index to the contribution of configuring this node with this
    # bitwidth to the minimal possible KPI of the model.
    layer_to_kpi_mapping, minimal_kpi = _compute_kpis(layer_to_bitwidth_mapping,
                                                      compute_kpi_fn)

    assert minimal_kpi.weights_memory <= target_kpi.weights_memory, f'Minimal KPI cannot be greater than target KPI. Minimal KPI:{minimal_kpi}, Target KPI:{target_kpi}'

    capacity = int(target_kpi.weights_memory - minimal_kpi.weights_memory)
    values = [[metric_value for _, metric_value in nodes_metrics_values.items()]
              for node_idx, nodes_metrics_values in layer_to_metrics_mapping.items()]
    weights = [[kpi_weight.weights_memory for _, kpi_weight in nodes_kpi_weights.items()]
               for node_idx, nodes_kpi_weights in layer_to_kpi_mapping.items()]

    # initialization of dynamic array and first level
    min_config = _get_minimal_size_configuration(layer_to_bitwidth_mapping)
    min_config_value = sum([values[layer_idx][min_config[layer_idx]] for layer_idx, _ in layer_to_bitwidth_mapping.items()])
    last = np.full((capacity + 1), {"value": min_config_value, "conf": min_config})  # TODO: value needs to be metric value of min_conf?
    for i in range(len(weights[0])):
        if weights[0][i] <= capacity:
            prev_capacity_idx = int(weights[0][i])
            prev_best_for_capacity = last[prev_capacity_idx]
            new_elem = _get_updated_config(prev_best_for_capacity, values[0], 0, i)
            last[int(weights[0][i])] = new_elem

    # updating table
    # current = np.empty(capacity + 1)
    for i in range(1, len(weights)):
        # for each class of items (layer)
        # current.fill(-np.inf)
        current = np.full((capacity + 1), {"value": min_config_value, "conf": min_config}) # TODO: value needs to be metric value of min_conf?
        for j in range(len(weights[i])):
            # for each item in the class (bitwidth)
            for k in range(int(weights[i][j]), capacity + 1):
                # go over all possible slots up to the maximal capacity and update
                # if the current item (bitwidth of layer i) can be used for a solution with the current capacity
                # (if gives better result than current result in that capacity)
                # TODO: maybe because our info on the problem we can skip unnecessary steps here,
                #  like stop after we exceed some capacity.
                #  requires to better understand the solution
                if last[(k - int(weights[i][j]))]["value"] > min_config_value:
                    # i is the layer that we currently work on
                    # j is the index of the bitwidth for the layer
                    # k is the current upper bound on the total size of the reduced model
                    new_elem = _get_updated_config(prev_best_for_capacity=last[(k - int(weights[i][j]))],
                                                  layer_values=values[i],
                                                  layer_idx=i,
                                                  bitwidth_idx=j)
                    current[k] = new_elem

        # temp = current
        # current = last
        # last = temp
        last = current

    # get best results with tie-breaker by higher precision (larger total weight)
    max_value = max(last, key=itemgetter('value'))['value']
    all_max_res = [res for res in last if res['value'] == max_value]
    max_res = max(all_max_res, key=lambda res: compute_kpi_fn(res['conf']).weights_memory)
    # max_res = max(last, key=itemgetter('value'))
    print(max_res["value"], max_res["conf"])
    # print(list(filter(lambda d: d['value'] != min_config_value, last)))
    return np.asarray(max_res["conf"])


def _build_layer_to_metrics_mapping(node_to_bitwidth_indices: Dict[int, List[int]],
                                    compute_metric_fn: Callable) -> Dict[int, Dict[int, float]]:
    """
    This function measures the sensitivity of a change in a bitwidth of a layer on the entire model.
    It builds a mapping from a node's index, to its bitwidht's effect on the model sensitivity.
    For each node and some possible node's bitwidth (according to the given search space), we use
    the framework function compute_metric_fn in order to infer
    a batch of images, and compute (using the inference results) the sensitivity metric of
    the configured mixed-precision model.

    Args:
        node_to_bitwidth_indices: Possible bitwidth indices for the different nodes.
        compute_metric_fn: Function to measure a sensitivity metric.

    Returns:
        Mapping from each node's index in a graph, to a dictionary from the bitwidth index (of this node) to
        the sensitivity of the model.

    """

    Logger.info('Starting to evaluate metrics')
    layer_to_metrics_mapping = {}

    for node_idx, layer_possible_bitwidths_indices in tqdm(node_to_bitwidth_indices.items(),
                                                           total=len(node_to_bitwidth_indices)):
        layer_to_metrics_mapping[node_idx] = {}

        for bitwidth_idx in layer_possible_bitwidths_indices:
            # Create a configuration that differs at one layer only from the baseline model
            mp_model_configuration = [0] * len(node_to_bitwidth_indices)
            mp_model_configuration[node_idx] = bitwidth_idx

            # Build a distance matrix using the function we got from the framework implementation.
            layer_to_metrics_mapping[node_idx][bitwidth_idx] = -1 * compute_metric_fn(mp_model_configuration,
                                                                                      [node_idx])

    return layer_to_metrics_mapping


def _compute_kpis(node_to_bitwidth_indices: Dict[int, List[int]],
                  compute_kpi_fn: Callable) -> Tuple[Dict[int, Dict[int, KPI]], KPI]:
    """
    This function computes and returns:
    1. The minimal possible KPI of the graph.
    2. A mapping from each node's index to a mapping from a possible bitwidth index to
    the contribution to the model's minimal KPI, if we were configuring this node with this bitwidth.

    Args:
        node_to_bitwidth_indices: Possible indices for the different nodes.
        compute_kpi_fn: Function to compute a mixed-precision model KPI for a given
        mixed-precision bitwidth configuration.

    Returns:
        A tuple containing a mapping from each node's index in a graph, to a dictionary from the
        bitwidth index (of this node) to the contribution to the minimal KPI of the model.
        The second element in the tuple is the minimal possible KPI.

    """

    Logger.info('Starting to compute KPIs per node and bitwidth')
    layer_to_kpi_mapping = {}

    minimal_graph_size_configuration = _get_minimal_size_configuration(node_to_bitwidth_indices)

    minimal_kpi = compute_kpi_fn(minimal_graph_size_configuration)  # minimal possible kpi

    for node_idx, layer_possible_bitwidths_indices in tqdm(node_to_bitwidth_indices.items(),
                                                           total=len(node_to_bitwidth_indices)):
        layer_to_kpi_mapping[node_idx] = {}
        for bitwidth_idx in layer_possible_bitwidths_indices:

            # Change the minimal KPI configuration at one node only and
            # compute this change's contribution to the model's KPI.
            mp_model_configuration = minimal_graph_size_configuration.copy()
            mp_model_configuration[node_idx] = bitwidth_idx

            mp_model_kpi = compute_kpi_fn(mp_model_configuration)
            contribution_to_minimal_model = mp_model_kpi.weights_memory - minimal_kpi.weights_memory

            layer_to_kpi_mapping[node_idx][bitwidth_idx] = KPI(contribution_to_minimal_model)

    return layer_to_kpi_mapping, minimal_kpi


def _get_minimal_size_configuration(node_to_bitwidth_indices):
    # The node's candidates are sorted in a descending order, thus we take the last index of each node.
    minimal_graph_size_configuration = [node_to_bitwidth_indices[node_idx][-1] for node_idx in
                                        sorted(node_to_bitwidth_indices.keys())]
    return minimal_graph_size_configuration


def _get_updated_config(prev_best_for_capacity, layer_values, layer_idx, bitwidth_idx):
    prev_best_value = prev_best_for_capacity['value']
    prev_best_cfg = prev_best_for_capacity['conf']
    new_value = prev_best_value - layer_values[prev_best_cfg[layer_idx]] + layer_values[bitwidth_idx]
    if prev_best_value > new_value:
        # no update
        return prev_best_for_capacity
    elif prev_best_value < new_value:
        # take new configuration (better value)
        return {"value": new_value,
                "conf": prev_best_cfg[:layer_idx] + [bitwidth_idx] + prev_best_cfg[layer_idx + 1:]}
    else:
        # values are equal, tie-breaker by maximal weight
        prev_layer_bitwidth_idx = prev_best_cfg[layer_idx]
        new_bitwidth_idx = min(bitwidth_idx, prev_layer_bitwidth_idx)
        return {"value": new_value,
                "conf": prev_best_cfg[:layer_idx] + [new_bitwidth_idx] + prev_best_cfg[layer_idx + 1:]}
