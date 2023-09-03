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
from typing import List

import os

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.visualization.final_config_visualizer import \
    WeightsFinalBitwidthConfigVisualizer, ActivationFinalBitwidthConfigVisualizer
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.logger import Logger


def init_tensorboard_writer(fw_info: FrameworkInfo) -> TensorboardWriter:
    """
    Create a TensorBoardWriter object initialized with the logger dir path if it was set,
    or None otherwise.

    Args:
        fw_info: FrameworkInfo object.

    Returns:
        A TensorBoardWriter object.
    """
    tb_w = None
    if Logger.LOG_PATH is not None:
        tb_log_dir = os.path.join(os.getcwd(), Logger.LOG_PATH, 'tensorboard_logs')
        Logger.info(f'To use Tensorboard, please run: tensorboard --logdir {tb_log_dir}')
        tb_w = TensorboardWriter(tb_log_dir, fw_info)
    return tb_w


def finalize_bitwidth_in_tb(tb_w: TensorboardWriter,
                            weights_conf_nodes_bitwidth: List,
                            activation_conf_nodes_bitwidth: List):
    if len(weights_conf_nodes_bitwidth) > 0:
        visual = WeightsFinalBitwidthConfigVisualizer(weights_conf_nodes_bitwidth)
        figure = visual.plot_config_bitwidth()
        tb_w.add_figure(figure, f'Weights final bit-width config')
    if len(activation_conf_nodes_bitwidth) > 0:
        visual = ActivationFinalBitwidthConfigVisualizer(activation_conf_nodes_bitwidth)
        figure = visual.plot_config_bitwidth()
        tb_w.add_figure(figure, f'Activation final bit-width config')