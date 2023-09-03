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
from typing import Callable, List

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_facade import search_bit_width
from model_compression_toolkit.logger import Logger


def mixed_precision_runner(graph: Graph,
                           representative_data_gen: Callable,
                           core_config: CoreConfig,
                           fw_info: FrameworkInfo,
                           fw_impl: FrameworkImplementation,
                           target_kpi: KPI) -> List[int]:

    if core_config.mixed_precision_config.configuration_overwrite is None:

        bit_width_config = search_bit_width(graph,
                                            fw_info,
                                            fw_impl,
                                            target_kpi,
                                            core_config.mixed_precision_config,
                                            representative_data_gen)
    else:
        Logger.warning(
            f'Mixed Precision has overwrite bit-width configuration{core_config.mixed_precision_config.configuration_overwrite}')
        bit_width_config = core_config.mixed_precision_config.configuration_overwrite

    return bit_width_config
