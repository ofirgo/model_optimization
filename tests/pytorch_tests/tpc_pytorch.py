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

import model_compression_toolkit as mct
from torch.nn import ConvTranspose2d


from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc
from model_compression_toolkit.core.pytorch.reader.node_holders import DummyPlaceHolder
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tpc

tp = mct.target_platform


def get_pytorch_test_tpc_dict(tp_model, test_name, ftp_name):
    return {
        test_name: generate_pytorch_tpc(name=ftp_name,
                                        tp_model=tp_model),
    }


def get_mp_activation_pytorch_tpc_dict(tpc_model, test_name, tpc_name):
    op_sets_to_layer_add = {
        "Input": [DummyPlaceHolder],
        "Conv": [ConvTranspose2d]
    }

    return {
        test_name: generate_test_tpc(name=tpc_name,
                                     tp_model=tpc_model,
                                     base_tpc=generate_pytorch_tpc(name=f"base_{tpc_name}",
                                                                   tp_model=tpc_model),
                                     op_sets_to_layer_add=op_sets_to_layer_add),
    }
