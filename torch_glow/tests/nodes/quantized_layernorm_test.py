# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class QuantizedLayerNormModule(torch.nn.Module):
    def __init__(self, normalized_shape, scale, zero_point, weight=None, bias=None):
        super(QuantizedLayerNormModule, self).__init__()
        self.normalized_shape = normalized_shape
        self.scale = scale
        self.zero_point = zero_point
        self.weight = weight
        self.bias = bias
        self.quant = torch.nn.quantized.Quantize(
            scale=0.3, zero_point=128, dtype=torch.quint8
        )

    def forward(self, x):
        x = self.quant(x)
        x = torch.ops.quantized.layer_norm(
            x,
            self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=1e-05,
            output_scale=self.scale,
            output_zero_point=self.zero_point,
        )
        return torch.dequantize(x)


class TestQuantizedLayerNorm(utils.TorchGlowTestCase):
    def test_layernorm_basic(self):
        """Basic test of the PyTorch quantized layernorm Node on Glow."""

        inputs = torch.tensor([0.3, 0.6, 0.3]).reshape(1, 1, 3)
        weight = torch.tensor([1.0, 1.1, 1.2])
        bias = torch.tensor([0.1, 0.1, 0.2])

        utils.compare_tracing_methods(
            QuantizedLayerNormModule([3], 0.01, 66, weight, bias),
            inputs,
            fusible_ops={"quantized::layer_norm"},
            atol=1e-02,
        )

    def test_layernorm_no_weight_bias(self):
        """Test of the PyTorch quantized::layer_norm without weights and bias."""

        inputs = torch.tensor([0.3, 0.6, 0.9, 0.3]).reshape(1, 1, 2, 2)

        utils.compare_tracing_methods(
            QuantizedLayerNormModule([2, 2], 0.01, 91),
            inputs,
            fusible_ops={"quantized::layer_norm"},
            atol=1e-2,
        )
