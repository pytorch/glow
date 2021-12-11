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


class SimpleQuantizedMulModel(torch.nn.Module):
    def __init__(
        self, left_quantization, right_quantization=None, scale=None, zero_point=None
    ):
        super(SimpleQuantizedMulModel, self).__init__()
        self.scale = scale
        self.zero_point = zero_point
        self.left_quantization = left_quantization
        self.right_quantization = right_quantization or left_quantization

    def forward(self, tensor, other):
        if other.size() == torch.Size([]):
            return torch.nn.quantized.DeQuantize()(
                torch.ops.quantized.mul(self.left_quantization(tensor), other.item())
            )
        else:
            return torch.nn.quantized.DeQuantize()(
                torch.ops.quantized.mul(
                    self.left_quantization(tensor),
                    self.right_quantization(other),
                    scale=self.scale,
                    zero_point=self.zero_point,
                )
            )


class TestQuantizedMul(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "zero_offset",
                SimpleQuantizedMulModel(
                    torch.nn.quantized.Quantize(
                        scale=0.3, zero_point=0, dtype=torch.quint8
                    ),
                    torch.nn.quantized.Quantize(
                        scale=0.3, zero_point=0, dtype=torch.quint8
                    ),
                    0.05,
                    0,
                ),
                torch.tensor([1, 2, 3, 4], dtype=torch.float32),
                torch.tensor([5, 6, 7, 8], dtype=torch.float32),
            ),
            lambda: (
                "basic",
                SimpleQuantizedMulModel(
                    torch.nn.quantized.Quantize(
                        scale=0.2, zero_point=1, dtype=torch.quint8
                    ),
                    torch.nn.quantized.Quantize(
                        scale=0.2, zero_point=1, dtype=torch.quint8
                    ),
                    0.2,
                    3,
                ),
                torch.randn([5, 5]),
                torch.randn([5, 5]),
            ),
            lambda: (
                "cut_q_dq",
                SimpleQuantizedMulModel(
                    torch.nn.quantized.Quantize(
                        scale=0.2, zero_point=5, dtype=torch.quint8
                    ),
                    torch.nn.quantized.Quantize(
                        scale=0.2, zero_point=10, dtype=torch.quint8
                    ),
                    0.2,
                    3,
                ),
                torch.randn([5, 5]),
                torch.randn([5, 5]),
                ["aten::quantize_per_tensor", "aten::dequantize"],
            ),
            lambda: (
                "broadcast_rhs",
                SimpleQuantizedMulModel(
                    torch.nn.quantized.Quantize(
                        scale=0.2, zero_point=1, dtype=torch.quint8
                    ),
                    torch.nn.quantized.Quantize(
                        scale=0.2, zero_point=1, dtype=torch.quint8
                    ),
                    0.2,
                    3,
                ),
                torch.randn([1, 5, 6, 6]),
                torch.randn([1, 5, 1, 1]),
                ["aten::quantize_per_tensor", "aten::dequantize"],
            ),
            lambda: (
                "broadcast_lhs",
                SimpleQuantizedMulModel(
                    torch.nn.quantized.Quantize(
                        scale=0.2, zero_point=1, dtype=torch.quint8
                    ),
                    torch.nn.quantized.Quantize(
                        scale=0.2, zero_point=1, dtype=torch.quint8
                    ),
                    0.2,
                    3,
                ),
                torch.randn([1, 5, 1, 1]),
                torch.randn([1, 5, 6, 6]),
                ["aten::quantize_per_tensor", "aten::dequantize"],
            ),
            lambda: (
                "broadcast_with_implicit_degenerate_dimension",
                SimpleQuantizedMulModel(
                    torch.nn.quantized.Quantize(
                        scale=0.2, zero_point=1, dtype=torch.quint8
                    ),
                    torch.nn.quantized.Quantize(
                        scale=0.2, zero_point=1, dtype=torch.quint8
                    ),
                    0.2,
                    3,
                ),
                torch.randn([1, 3, 1, 1]),
                torch.randn([3, 4, 4]),
                ["aten::quantize_per_tensor", "aten::dequantize"],
            ),
            lambda: (
                "broadcast_outer",
                SimpleQuantizedMulModel(
                    torch.nn.quantized.Quantize(
                        scale=0.2, zero_point=1, dtype=torch.quint8
                    ),
                    torch.nn.quantized.Quantize(
                        scale=0.2, zero_point=1, dtype=torch.quint8
                    ),
                    0.2,
                    3,
                ),
                torch.randn([1, 5]),
                torch.randn([6, 1]),
                ["aten::quantize_per_tensor", "aten::dequantize"],
            ),
            lambda: (
                "positive_scalar",
                SimpleQuantizedMulModel(
                    torch.nn.quantized.Quantize(
                        scale=0.05, zero_point=1, dtype=torch.quint8
                    ),
                ),
                torch.randn(1, 2, 3, 4),
                torch.tensor(3.14),
            ),
            lambda: (
                "negative_scalar",
                SimpleQuantizedMulModel(
                    torch.nn.quantized.Quantize(
                        scale=0.05, zero_point=2, dtype=torch.quint8
                    ),
                ),
                torch.randn(1, 2, 3, 4),
                torch.tensor(-3.14),
            ),
            lambda: (
                "zero_scalar",
                SimpleQuantizedMulModel(
                    torch.nn.quantized.Quantize(
                        scale=0.05, zero_point=2, dtype=torch.quint8
                    ),
                ),
                torch.randn(1, 2, 3, 4),
                torch.tensor(0.00),
            ),
            lambda: (
                "negative_int8_scalar",
                SimpleQuantizedMulModel(
                    torch.nn.quantized.Quantize(
                        scale=0.05, zero_point=4, dtype=torch.qint8
                    ),
                ),
                torch.randn(1, 2, 3, 4),
                torch.tensor(-1.43),
            ),
        ]
    )
    def test_quantized_mul(self, _, module, tensor, other, fusion_blocklist=None):
        utils.compare_tracing_methods(
            module,
            tensor,
            other,
            fusible_ops={"quantized::mul"},
            fusion_blocklist=fusion_blocklist,
            skip_to_glow=True,
        )
