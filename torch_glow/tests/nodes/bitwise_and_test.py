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


class SimpleBitwiseAndModule(torch.nn.Module):
    def __init__(self, dtype=None):
        super(SimpleBitwiseAndModule, self).__init__()
        self.dtype = dtype

    def forward(self, a, b):
        c = torch.bitwise_and(a, b)
        d = torch.bitwise_and(a, c)
        return d


class TestBitwiseAnd(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "basic",
                torch.tensor([0x01, 0x02, 0x03, 0x04], dtype=torch.int32),
                torch.tensor([0x03, 0x07, 0xFF, 0x0100], dtype=torch.int32),
            ),
            lambda: (
                "basic_64",
                torch.tensor([0x01, 0x03, 0xFFFFFFFFFF00, 0x5], dtype=torch.int64),
                torch.tensor([0x02, 0x03, 0x2, 0x1F], dtype=torch.int64),
            ),
            lambda: (
                "basic_bool",
                torch.tensor([True, True, False, False], dtype=torch.bool),
                torch.tensor([True, False, True, False], dtype=torch.bool),
            ),
            lambda: (
                "basic_3d",
                torch.zeros((0x2, 0x1, 0x5), dtype=torch.int32),
                torch.ones((0x1, 0x6, 0x5), dtype=torch.int32),
            ),
            lambda: (
                "broadcast_3d",
                torch.zeros((3, 4, 5), dtype=torch.int32),
                torch.ones((4, 5), dtype=torch.int32),
            ),
        ]
    )
    def test_bitwise_and(self, _, a, b, skip_to_glow=False):
        """Tests of the PyTorch Bitwise And Node on Glow."""
        utils.compare_tracing_methods(
            SimpleBitwiseAndModule(),
            a,
            b,
            fusible_ops={"aten::bitwise_and"},
        )
