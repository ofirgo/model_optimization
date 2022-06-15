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
from enum import Enum
from functools import partial

import numpy as np


class CorrelationMeasure(Enum):
    LINEAR_CKA_FEATURES = "LINEAR_CKA_FEATURES",
    LINEAR_CKA_EXAMPLES = "LINEAR_CKA_EXAMPLES",
    RBF_CKA = "RBF_CKA",
    CCA_CORR = "CCA_CORR",


def _flatten_tensor_features(t: np.ndarray) -> np.ndarray:
    return np.reshape(t, newshape=(t.shape[0], -1))


def _center_tensor_features(t: np.ndarray) -> np.ndarray:
    return t - np.mean(t, 0, keepdims=True)


def _center_tensor_examples(t: np.ndarray) -> np.ndarray:
    if not np.allclose(t, t.T):
        raise ValueError('Input must be a symmetric matrix.')
    _t = t.copy().astype(np.float64)

    # TODO: working according to biased estimate of HSIC
    means = np.mean(_t, axis=0) # dtype=np.float64
    means -= np.mean(means) / 2
    _t -= means[:, None]  # means as column vector
    _t -= means[None, :]  # means as row vector
    return _t


def _get_rbf_examples_corr(x: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    x_dot_products = x.dot(x.T)
    sq_norms = np.diag(x_dot_products)
    sq_distances = -2 * x_dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def _cka(x_corr, y_corr):
    # Center the examples matrices
    x_corr_centered = _center_tensor_examples(x_corr)
    y_corr_centered = _center_tensor_examples(y_corr)

    scaled_hsic = x_corr_centered.ravel().dot(y_corr_centered.ravel())

    normalization_x = np.linalg.norm(x_corr_centered)
    normalization_y = np.linalg.norm(y_corr_centered)
    return scaled_hsic / (normalization_x * normalization_y)


def linear_cka_corr_from_features(x: np.ndarray, y: np.ndarray) -> float:

    # Flattening all tensors but the first axis (batch dimension)
    x_r = _flatten_tensor_features(x)
    y_r = _flatten_tensor_features(y)

    # Centring the features across all samples
    x_c = _center_tensor_features(x_r)
    y_c = _center_tensor_features(y_r)

    dot_similarity = np.linalg.norm(x_c.T.dot(y_c), ord='fro') ** 2
    x_norm = np.linalg.norm(x_c.T.dot(x_c), ord='fro')
    y_norm = np.linalg.norm(y_c.T.dot(y_c), ord='fro')

    return dot_similarity / (x_norm * y_norm)


def linear_cka_corr_from_examples(x: np.ndarray, y: np.ndarray) -> float:

    # Flattening all tensors but the first axis (batch dimension)
    x_r = _flatten_tensor_features(x)
    y_r = _flatten_tensor_features(y)

    # Get examples linear correlation matrices
    x_corr = x_r.dot(x_r.T)
    y_corr = y_r.dot(y_r.T)

    return _cka(x_corr, y_corr)


def rbf_cka_corr(x: np.ndarray, y: np.ndarray, threshold: float = 0.8) -> float:

    # Flattening all tensors but the first axis (batch dimension)
    x_r = _flatten_tensor_features(x)
    y_r = _flatten_tensor_features(y)

    # Get examples RBF correlation matrices
    x_corr = _get_rbf_examples_corr(x_r, threshold)
    y_corr = _get_rbf_examples_corr(y_r, threshold)

    return _cka(x_corr, y_corr)


def cca_corr(x: np.ndarray, y: np.ndarray) -> float:
    # Mean squared CCA correlation

    # Flattening all tensors but the first axis (batch dimension)
    x_r = _flatten_tensor_features(x)
    y_r = _flatten_tensor_features(y)

    qx, _ = np.linalg.qr(x_r)
    qy, _ = np.linalg.qr(y_r)

    # TODO: do we need to center the tensors?
    return np.linalg.norm(qx.T.dot(qy)) ** 2 / min(x_r.shape[1], y_r.shape[1])
