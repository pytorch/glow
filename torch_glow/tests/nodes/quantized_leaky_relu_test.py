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


class SimpleQuantizedLeakyReluModel(torch.nn.Module):
    def __init__(self, scale, zero_point, dtype):
        super(SimpleQuantizedLeakyReluModel, self).__init__()
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    def forward(self, tensor):
        quantize = torch.nn.quantized.Quantize(
            scale=self.scale, zero_point=self.zero_point, dtype=self.dtype
        )
        dequantize = torch.nn.quantized.DeQuantize()
        leaky_relu = torch.nn.LeakyReLU()
        return dequantize(leaky_relu(quantize(tensor)))


class TestQuantizedLeakyRelu(utils.TorchGlowTestCase):
    def test_quantized_leaky_relu(self):
        """Basic test of the PyTorch quantized::leaky_relu Node on Glow."""

        utils.compare_tracing_methods(
            SimpleQuantizedLeakyReluModel(0.3, 0, torch.quint8),
            torch.randn([5, 5]),
            fusible_ops={
                "aten::leaky_relu",
                "aten::quantize_per_tensor",
                "aten::dequantize",
            },
        )

    def test_quantized_leaky_relu_cut_dq(self):
        """Basic test of the PyTorch quantized::leaky_relu Node on Glow, with quantize and dequantize excluded."""

        utils.compare_tracing_methods(
            SimpleQuantizedLeakyReluModel(0.3, 0, torch.quint8),
            torch.randn([5, 5]),
            fusible_ops={"aten::leaky_relu", "aten::quantize_per_tensor"},
            fusion_blocklist=["aten::dequantize"],
        )
