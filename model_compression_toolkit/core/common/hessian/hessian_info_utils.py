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
from typing import List
import numpy as np
from model_compression_toolkit.constants import EPS


def normalize_weights(trace_hessian_approximations: List) -> List[float]:
    """
    Normalize trace Hessian approximations by dividing the trace Hessian approximations value by the sum of all
    other values.

    Args:
        trace_hessian_approximations: Approximated average jacobian-based weights for each interest point.

    Returns:
            Normalized list of trace Hessian approximations for each interest point.
    """
    scores_vec = np.asarray(trace_hessian_approximations)

    return (scores_vec / (np.sum(scores_vec) + EPS)).tolist()