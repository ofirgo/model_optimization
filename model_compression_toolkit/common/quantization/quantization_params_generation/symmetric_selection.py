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
from scipy import optimize

import model_compression_toolkit.common.quantization.quantization_config as qc
from model_compression_toolkit.common.constants import MIN_THRESHOLD, THRESHOLD
from model_compression_toolkit.common.quantization.quantization_params_generation.kl_selection import \
    _kl_error_histogram, _kl_batch_error_function
from model_compression_toolkit.common.quantization.quantization_params_generation.lp_selection import \
    _lp_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.mae_selection import \
    _mae_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.mse_selection import \
    _mse_error_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_search import \
    qparams_histogram_minimization, \
    symmetric_quantization_loss, kl_symmetric_qparams_histogram_minimization, kl_symmetric_quantization_loss
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import \
    get_tensor_max, quantize_tensor, get_output_shape

from model_compression_toolkit.common.similarity_analyzer import compute_mse, compute_mae, compute_lp_norm


def symmetric_selection_tensor(tensor_data: np.ndarray,
                               p: int,
                               n_bits: int,
                               per_channel: bool = False,
                               channel_axis: int = 1,
                               n_iter: int = 10,
                               min_threshold: float = MIN_THRESHOLD,
                               quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.MSE) -> dict:
    """
    Compute the optimal threshold based on the provided QuantizationErrorMethod to quantize the tensor.
    Different search is applied, depends on the value of the selected QuantizationErrorMethod.

    Args:
        tensor_data: Tensor content as Numpy array.
        p: p-norm to use for the Lp-norm distance.
        n_bits: Number of bits to quantize the tensor.
        per_channel: Whether the quantization should be per-channel or not.
        channel_axis: Output channel index.
        n_iter: Number of iterations to search for the optimal threshold (not used for this method).
        min_threshold: Minimal threshold to use if threshold is too small (not used for this method).
        quant_error_method: an error function to optimize the parameters' selection accordingly.

    Returns:
        Optimal threshold to quantize the tensor in a symmetric manner.
    """

    signed = np.any(tensor_data < 0)  # check if tensor is singed
    unsigned_tensor_data = np.abs(tensor_data)
    tensor_max = get_tensor_max(unsigned_tensor_data, per_channel, channel_axis)

    if quant_error_method == qc.QuantizationErrorMethod.NOCLIPPING:
        res = tensor_max
    elif quant_error_method == qc.QuantizationErrorMethod.KL:
        # search for KL error is separated because the error method signature is different from the other error methods.
        # we use _kl_batch_error_function to allow calculation per_channel in a vectorized manner if necessary,
        # we pass it as argument to avoid exposing protected package member inside kl_symmetric_quantization_loss.
        error_wrapper = lambda threshold: kl_symmetric_quantization_loss(_kl_batch_error_function,
                                                                         tensor_data,
                                                                         threshold,
                                                                         signed,
                                                                         per_channel,
                                                                         channel_axis)
        res = optimize.minimize(fun=error_wrapper, x0=tensor_max, jac=lambda x: loss_function_jac(x, f=error_wrapper))

        # res.x contains the actual optimized parameters result from optimize.minimize
        res = res.x if not per_channel else \
            np.reshape(res.x, get_output_shape(tensor_data.shape, channel_axis))
    else:
        error_function = get_threshold_selection_tensor_error_function(quant_error_method, p)
        error_wrapper = lambda threshold: symmetric_quantization_loss(error_function,
                                                                      tensor_data,
                                                                      threshold,
                                                                      n_bits,
                                                                      signed,
                                                                      per_channel,
                                                                      channel_axis)
        res = optimize.minimize(fun=error_wrapper, x0=tensor_max, jac=lambda x: loss_function_jac(x, f=error_wrapper))

        # res.x contains the actual optimized parameters result from optimize.minimize
        res = res.x if not per_channel else \
            np.reshape(res.x, get_output_shape(tensor_data.shape, channel_axis))

    return {THRESHOLD: res}


def loss_function_jac(x, f):
    f0 = f(x)
    rel_step = _eps_for_method(x.dtype, f0.dtype)
    sign_x0 = (x >= 0).astype(float) * 2 - 1
    h = rel_step * sign_x0 * np.maximum(1.0, np.abs(x))

    x_next = x + h
    f_next = f(x_next)

    df = f_next - f0
    dx = x_next - x
    J_transposed = df / dx
    J_transposed = np.ravel(J_transposed)
    return J_transposed


def _eps_for_method(x0_dtype, f0_dtype):
    # the default EPS value
    EPS = np.finfo(np.float64).eps

    x0_is_fp = False
    if np.issubdtype(x0_dtype, np.inexact):
        # if you're a floating point type then over-ride the default EPS
        EPS = np.finfo(x0_dtype).eps
        x0_itemsize = np.dtype(x0_dtype).itemsize
        x0_is_fp = True

    if np.issubdtype(f0_dtype, np.inexact):
        f0_itemsize = np.dtype(f0_dtype).itemsize
        # choose the smallest itemsize between x0 and f0
        if x0_is_fp and f0_itemsize < x0_itemsize:
            EPS = np.finfo(f0_dtype).eps

    return EPS ** 0.5


def symmetric_selection_histogram(bins: np.ndarray,
                                  counts: np.ndarray,
                                  p: int,
                                  n_bits: int,
                                  min_value: float,
                                  max_value: float,
                                  constrained: bool = True,
                                  n_iter: int = 10,
                                  min_threshold: float = MIN_THRESHOLD,
                                  quant_error_method: qc.QuantizationErrorMethod = qc.QuantizationErrorMethod.MSE) -> dict:
    """
    Compute the optimal threshold based on the provided QuantizationErrorMethod to quantize a histogram.
    Different search is applied, depends on the value of the selected QuantizationErrorMethod.

    Args:
        bins: Bins values of the histogram.
        counts: Bins counts of the histogram.
        p: p-norm to use for the Lp-norm distance (used only for lp threshold selection).
        n_bits: Number of bits to quantize the tensor.
        min_value: Min value (not used for this method).
        max_value: Max value (not used for this method).
        constrained: Whether the threshold should be constrained or not (not used for this method).
        n_iter: Number of iteration ot search for the threshold (not used for this method).
        min_threshold: Minimal threshold to use if threshold is too small (used only for kl threshold selection).
        quant_error_method: an error function to optimize the parameters' selection accordingly.

    Returns:
        Optimal threshold to quantize the histogram a symmetric manner.
    """
    tensor_max = np.max(np.abs(bins))
    signed = np.any(bins < 0)  # check if tensor is singed
    if quant_error_method == qc.QuantizationErrorMethod.NOCLIPPING:
        res = tensor_max
    elif quant_error_method == qc.QuantizationErrorMethod.KL:
        # search for KL error is separated because the error method signature is different from the other error methods.
        # we pass it as argument to avoid exposing protected package member inside kl_symmetric_quantization_loss.
        res = kl_symmetric_qparams_histogram_minimization(bins,
                                                          tensor_max,
                                                          counts,
                                                          n_bits,
                                                          signed,
                                                          error_function=_kl_error_histogram)
        # res.x contains the actual optimized parameters result from optimize.minimize.
        # It is a vector with single element, therefore, we are taking res.x[0]
        res = res.x[0]

    else:
        error_function = get_threshold_selection_histogram_error_function(quant_error_method, p)
        res = qparams_histogram_minimization(bins,
                                             tensor_max,
                                             counts,
                                             error_function=error_function,
                                             quant_function=lambda threshold:
                                             quantize_tensor(bins, threshold, n_bits, signed))

        # res.x contains the actual optimized parameters result from optimize.minimize.
        # It is a vector with single element, therefore, we are taking res.x[0].
        res = res.x[0]
    return {THRESHOLD: res}


def get_threshold_selection_tensor_error_function(quant_error_method, p):
    """
    Returns the error function compatible to the provided threshold method,
    to be used in the threshold optimization search for tensor quantization.
    Args:
        quant_error_method: the requested error function type.
        p: p-norm to use for the Lp-norm distance.


    Returns: a Callable method that calculates the error between a tensor and a quantized tensor.

    """
    quant_method_error_function_mapping = {
        qc.QuantizationErrorMethod.MSE: compute_mse,
        qc.QuantizationErrorMethod.MAE: compute_mae,
        qc.QuantizationErrorMethod.LP: lambda x, y: compute_lp_norm(x, y, p),
    }

    return quant_method_error_function_mapping[quant_error_method]


def get_threshold_selection_histogram_error_function(quant_error_method, p):
    """
    Returns the error function compatible to the provided threshold method,
    to be used in the threshold optimization search for histogram quantization.
    Args:
        quant_error_method: the requested error function type.
        p: p-norm to use for the Lp-norm distance.

    Returns: a Callable method that calculates the error between a tensor and a quantized tensor.

    """
    quant_method_error_function_mapping = {
        qc.QuantizationErrorMethod.MSE: lambda q_bins, q_count, bins, counts, threshold:
            _mse_error_histogram(q_bins, q_count, bins, counts),
        qc.QuantizationErrorMethod.MAE: lambda q_bins, q_count, bins, counts, threshold:
            _mae_error_histogram(q_bins, q_count, bins, counts),
        qc.QuantizationErrorMethod.LP: lambda q_bins, q_count, bins, counts, threshold:
            _lp_error_histogram(q_bins, q_count, bins, counts, p=p),
    }

    return quant_method_error_function_mapping[quant_error_method]
