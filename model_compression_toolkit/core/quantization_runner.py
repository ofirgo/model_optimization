# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
import copy

from typing import Callable, List, Dict, Tuple
import numpy as np

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.mixed_precision.bit_width_setter import set_bit_widths
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPITarget
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_aggregation_methods import MpKpiAggregation
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_functions_mapping import kpi_functions_mapping
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_methods import MpKpiMetric
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_runner import mixed_precision_runner
from model_compression_toolkit.core.common.network_editors.edit_network import edit_network_graph
from model_compression_toolkit.core.common.visualization.tensorboard_util import finalize_bitwidth_in_tb
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.logger import Logger


def quantization_runner(graph: Graph,
                        representative_data_gen: Callable,
                        core_config: CoreConfig,
                        fw_info: FrameworkInfo,
                        fw_impl: FrameworkImplementation,
                        target_kpi: KPI = None,
                        tb_w: TensorboardWriter = None):

    # TODO: check whether to run mixed precision according to core_config.mixed_precision_enable only,
    #  enforce target_kpi with an assert.
    #  Later, possibly decide whether to use MP or not based on PTQ/Advanced PTQ facade or flag.
    #  Remember to also update condition in creation of bit_width_config (at the end of this method).
    if target_kpi is not None:
        assert core_config.mixed_precision_enable
        bit_width_config = mixed_precision_runner(graph,
                                                  representative_data_gen,
                                                  core_config,
                                                  fw_info,
                                                  fw_impl,
                                                  target_kpi)

    else:
        bit_width_config = []

    set_bit_widths(core_config.mixed_precision_enable,
                   graph,
                   bit_width_config)

    # Edit the graph after finalizing the configurations.
    # This is since some actions regard the final configuration and should be edited.
    edit_network_graph(graph, fw_info, core_config.debug_config.network_editor)

    _set_final_kpi(graph=graph,
                   kpi_functions_dict=kpi_functions_mapping,
                   fw_info=fw_info,
                   fw_impl=fw_impl,
                   final_bit_widths_config=bit_width_config)

    if target_kpi is not None:
        # Retrieve lists of tuples (node, node's final weights/activation bitwidth)
        weights_conf_nodes_bitwidth = graph.get_final_weights_config()
        activation_conf_nodes_bitwidth = graph.get_final_activation_config()

        Logger.info(
            f'Final weights bit-width configuration: {[node_b[1] for node_b in weights_conf_nodes_bitwidth]}')
        Logger.info(
            f'Final activation bit-width configuration: {[node_b[1] for node_b in activation_conf_nodes_bitwidth]}')

        if tb_w is not None:
            finalize_bitwidth_in_tb(tb_w, weights_conf_nodes_bitwidth, activation_conf_nodes_bitwidth)

    return graph, bit_width_config


def _set_final_kpi(graph: Graph,
                   kpi_functions_dict: Dict[KPITarget, Tuple[MpKpiMetric, MpKpiAggregation]],
                   fw_info: FrameworkInfo,
                   fw_impl: FrameworkImplementation,
                   final_bit_widths_config: List[int] = None):
    """
    Computing the KPIs of the model according to the final bit-width configuration,
    and setting it (inplace) in the graph's UserInfo field.

    Args:
        graph: Graph to compute the KPI for.
        kpi_functions_dict: A mapping between a KPITarget and a pair of kpi method and kpi aggregation functions.
        fw_info: A FrameworkInfo object.
        fw_impl: FrameworkImplementation object with specific framework methods implementation.
        final_bit_widths_config: The final bit-width configuration to quantize the model accordingly (relevant only in mixed-precision mode).

    """

    final_kpis_dict = {}
    for kpi_target, kpi_funcs in kpi_functions_dict.items():
        kpi_method, kpi_aggr = kpi_funcs
        if kpi_target == KPITarget.BOPS:
            final_kpis_dict[kpi_target] = \
            kpi_aggr(kpi_method(final_bit_widths_config, graph, fw_info, fw_impl, False), False)[0]
        else:
            non_conf_kpi = kpi_method([], graph, fw_info, fw_impl)
            conf_kpi = kpi_method(final_bit_widths_config, graph, fw_info, fw_impl)
            if len(final_bit_widths_config) > 0 and len(non_conf_kpi) > 0:
                final_kpis_dict[kpi_target] = kpi_aggr(np.concatenate([conf_kpi, non_conf_kpi]), False)[0]
            elif len(final_bit_widths_config) > 0 and len(non_conf_kpi) == 0:
                final_kpis_dict[kpi_target] = kpi_aggr(conf_kpi, False)[0]
            elif final_bit_widths_config is None and len(non_conf_kpi) > 0:
                # final_bit_widths_config == 0 ==> no configurable nodes,
                # thus, KPI can be computed from non_conf_kpi alone
                final_kpis_dict[kpi_target] = kpi_aggr(non_conf_kpi, False)[0]
            else:
                # No relevant nodes have been quantized with affect on the given target - since we only consider
                # in the model's final size the quantized layers size, this means that the final size for this target
                # is zero.
                Logger.warning(f"No relevant quantized layers for the KPI target {kpi_target} were found, the recorded"
                               f"final KPI for this target would be 0.")
                final_kpis_dict[kpi_target] = 0

    final_kpi = KPI()
    final_kpi.set_kpi_by_target(final_kpis_dict)
    graph.user_info.final_kpi = final_kpi
    graph.user_info.mixed_precision_cfg = final_bit_widths_config