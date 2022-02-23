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
import unittest

from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_search import \
    qparams_tensor_minimization, get_channel_clipping_noise, get_channel_rounding_noise
from model_compression_toolkit.common.quantization.quantizers.quantizers_helpers import calculate_delta, \
    quantize_tensor, uniform_quantize_tensor, get_threshold_bounds, get_range_bounds
from model_compression_toolkit.common.similarity_analyzer import compute_mse


class TestMSE(unittest.TestCase):

    def test_mse_uniform_symmetric(self):
        # UNIFORM
            # block_14_depthwise_bn
            # 0:
                # [ 0.34029305 -0.5112379  -0.46685207  1.145637   -0.13777626 -0.867706,  0.36445442 -0.6586418  -0.49969545]
                # min: -0.867706
                # max: 1.145637
                # bounds: [(-1.7354120016098022, 2.291274070739746), (-1.7354120016098022, 2.291274070739746)]
                # res: array([-0.87723572,  1.19088875])
                # CN_j: nan
                # RN_j: 3.354069e-06
                # delta_j: 0.00811029206091954
            # 2:
                # [-0.16928157 -1.059786   -0.11679491 -0.38866705 -0.08600968 -0.36160746,  0.561184    1.0822785   0.46205038]
                # min: -1.059786
                # max: 1.0822785
                # bounds: [(-2.1195719242095947, 2.1645569801330566), (-2.1195719242095947, 2.1645569801330566)]
                # res: array([-1.05821107,  1.08408036])
                # CN_j: 1.5423713e-06
                # RN_j: 2.0381149e-06
                # delta_j: 0.008401142839305768
            # block_2_depthwise_bn
            # 8:
                # [-0.3328705   0.32366857  0.09579693  0.23262033  0.91762894  0.2553501, -0.03073878  0.05441648 -0.5353873 ]
                # min: -0.5353873
                # max: 0.91762894
                # bounds: [(-1.0707745552062988, 1.835257887840271), (-1.0707745552062988, 1.835257887840271)]
                # res: [-0.53613156  0.91670828]
                # CN_j: 1.195545e-07
                # RN_j: 2.7263964e-06
                # delta_j: 0.005697411167428132
        # SYMMETRIC
            # block_14_depthwise_bn
            # 0:
                # [ 0.34029305 -0.5112379  -0.46685207  1.145637   -0.13777626 -0.867706,  0.36445442 -0.6586418  -0.49969545]
                # channel_threshold: 1.145637
                # bounds: [(3.725290298461914e-09, 2.291274070739746)]
                # res: [array([1.17070412])] --> -1.17070412, 1.16155799
                # CN_j: nan
                # RN_j: 2.785859244398055e-06
                # delta_j: [0.00914613]
                # signed: True
                # min_threshold: 3.725290298461914e-09
            # 2:
                # [-0.16928157 -1.059786   -0.11679491 -0.38866705 -0.08600968 -0.36160746,  0.561184    1.0822785   0.46205038]
                # channel_threshold: 1.0822785
                # bounds: [(3.725290298461914e-09, 2.1645569801330566)]
                # res: [array([1.13717344])] --> -1.13717344, 1.12828927
                # CN_j: nan
                # RN_j: 3.696288274048791e-06
                # delta_j: array([0.00888417])
                # signed: True
                # min_threshold: 3.725290298461914e-09
            # block_2_depthwise_bn
            # 8:
                # [-0.3328705   0.32366857  0.09579693  0.23262033  0.91762894  0.2553501, -0.03073878  0.05441648 -0.5353873 ]
                # channel_threshold: 0.91762894
                # bounds: [(3.725290298461914e-09, 1.835257887840271)]
                # res: [0.96349773]
                # CN_j: nan
                # RN_j: 1.4054733823400777e-06
                # delta_j: 0.00752732600965513
                # signed: True
                # min_threshold: 3.725290298461914e-09
        tensor = np.asarray([-0.16928157, -1.059786 ,  -0.11679491, -0.38866705, -0.08600968, -0.36160746,  0.561184,    1.0822785,   0.46205038]) ##
        tensor_max = 1.0822785 ##
        tensor_min = -1.059786 ##
        n_bits = 8
        signed = True ##
        min_threshold = 3.725290298461914e-09 ##

        symmetric_x0 = max(min_threshold, tensor_max)
        symmetric_bounds = get_threshold_bounds(min_threshold, symmetric_x0)
        symmetric_threshold = qparams_tensor_minimization(tensor,
                                                          symmetric_x0,
                                                          error_function=lambda x, y, t: compute_mse(x, y),
                                                          quant_function=lambda threshold:
                                                              quantize_tensor(tensor, threshold, n_bits, signed),
                                                          bounds=symmetric_bounds)

        s_CN_j, delta_j = get_channel_clipping_noise(tensor, n_bits, signed, threshold=symmetric_threshold.x)
        s_RN_j = get_channel_rounding_noise(tensor, n_bits, signed, threshold=symmetric_threshold.x)
        norm_s_CN_j = np.sqrt(s_CN_j) / delta_j
        norm_s_RN_j = np.sqrt(s_RN_j) / delta_j

        uniform_x0 = np.array([tensor_min, tensor_max])
        uniform_bounds = get_range_bounds(tensor_min, tensor_max)
        uniform_range = qparams_tensor_minimization(tensor,
                                                    uniform_x0,
                                                    error_function=lambda x, y, t: compute_mse(x, y),
                                                    quant_function=lambda min_max_range:
                                                        uniform_quantize_tensor(tensor,
                                                                                min_max_range[0],
                                                                                min_max_range[1],
                                                                                n_bits),
                                                    bounds=uniform_bounds)

        u_CN_j, delta_j = get_channel_clipping_noise(tensor, n_bits, a=uniform_range.x[0], b=uniform_range.x[1])
        u_RN_j = get_channel_rounding_noise(tensor, n_bits, a=uniform_range.x[0], b=uniform_range.x[1])
        norm_u_CN_j = np.sqrt(u_CN_j) / delta_j
        norm_u_RN_j = np.sqrt(u_RN_j) / delta_j

        print(s_CN_j, u_CN_j, s_RN_j, u_RN_j)


if __name__ == '__main__':
    unittest.main()
