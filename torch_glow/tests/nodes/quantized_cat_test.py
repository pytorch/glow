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


class SimpleQuantizedCatModel(torch.nn.Module):
    def __init__(self, dimension, scale, zero_point):
        super(SimpleQuantizedCatModel, self).__init__()
        self.dimension = dimension
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, a, b):
        return torch.nn.quantized.DeQuantize()(
            torch.ops.quantized.cat(
                (a, b),
                dim=self.dimension,
                scale=self.scale,
                zero_point=self.zero_point,
            )
        )


class TestQuantizedCat(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "zero_offset",
                SimpleQuantizedCatModel(
                    0,
                    0.05,
                    0,
                ),
                (
                    torch.nn.quantized.Quantize(
                        scale=0.3,
                        zero_point=0,
                        dtype=torch.quint8,
                    )(torch.randn([1, 2, 3, 4], dtype=torch.float32)),
                    torch.nn.quantized.Quantize(
                        scale=0.3,
                        zero_point=0,
                        dtype=torch.quint8,
                    )(torch.randn([5, 2, 3, 4], dtype=torch.float32)),
                ),
            ),
            lambda: (
                "basic",
                SimpleQuantizedCatModel(
                    1,
                    0.05,
                    0,
                ),
                (
                    torch.nn.quantized.Quantize(
                        scale=0.3,
                        zero_point=0.3,
                        dtype=torch.quint8,
                    )(torch.randn([8, 8, 8, 8], dtype=torch.float32)),
                    torch.nn.quantized.Quantize(
                        scale=0.3,
                        zero_point=0.3,
                        dtype=torch.quint8,
                    )(torch.randn([8, 8, 8, 8], dtype=torch.float32)),
                ),
            ),
            lambda: (
                "with_empty_tensor",
                SimpleQuantizedCatModel(
                    0,
                    0.05,
                    0,
                ),
                (
                    torch.nn.quantized.Quantize(
                        scale=0.2,
                        zero_point=0.1,
                        dtype=torch.quint8,
                    )(torch.empty(0, dtype=torch.float32)),
                    torch.nn.quantized.Quantize(
                        scale=0.2,
                        zero_point=0.1,
                        dtype=torch.quint8,
                    )(torch.randn([8, 8], dtype=torch.float32)),
                ),
            ),
            lambda: (
                "with_differing_quantizations",
                SimpleQuantizedCatModel(
                    2,
                    0.05,
                    0,
                ),
                (
                    torch.nn.quantized.Quantize(
                        scale=0.6,
                        zero_point=0.2,
                        dtype=torch.quint8,
                    )(torch.randn([7, 7, 7], dtype=torch.float32)),
                    torch.nn.quantized.Quantize(
                        scale=0.2,
                        zero_point=0.1,
                        dtype=torch.quint8,
                    )(torch.randn([7, 7, 7], dtype=torch.float32)),
                ),
            ),
        ]
    )
    def test_quantized_cat(self, _, module, tensors, fusion_blocklist=None):
        utils.compare_tracing_methods(
            module,
            *tensors,
            fusible_ops={"quantized::cat"},
            fusion_blocklist=None,
            skip_to_glow=False,
        )
