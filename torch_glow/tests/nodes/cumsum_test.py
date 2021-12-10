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


class SimpleCumSumModule(torch.nn.Module):
    def __init__(self, dim):
        super(SimpleCumSumModule, self).__init__()
        self.dim = dim

    def forward(self, tensor):
        return torch.cumsum(tensor, self.dim)


class TestCumSum(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("1", torch.randn(1), 0),
            lambda: ("2", torch.randn(2), 0),
            lambda: ("20", torch.randn(20), 0),
            lambda: ("3x4_0", torch.randn(3, 4), 0),
            lambda: ("3x4_1", torch.randn(3, 4), 1),
            lambda: ("3x4_-1", torch.randn(3, 4), -1),
            lambda: ("3x4_-2", torch.randn(3, 4), -2),
            lambda: ("3x4x5_0", torch.randn(3, 4, 5), 0),
            lambda: ("3x4x5_1", torch.randn(3, 4, 5), 1),
            lambda: ("3x4x5_2", torch.randn(3, 4, 5), 2),
            lambda: ("3x4x5_-1", torch.randn(3, 4, 5), -1),
            lambda: ("3x4x5_-2", torch.randn(3, 4, 5), -2),
            lambda: ("3x4x5_-3", torch.randn(3, 4, 5), -3),
            lambda: ("6x5x4x3_0", torch.randn(6, 5, 4, 3), 0),
            lambda: ("6x5x4x3_1", torch.randn(6, 5, 4, 3), 1),
            lambda: ("6x5x4x3_2", torch.randn(6, 5, 4, 3), 2),
            lambda: ("6x5x4x3_3", torch.randn(6, 5, 4, 3), 3),
            lambda: ("6x5x4x3_-1", torch.randn(6, 5, 4, 3), -1),
            lambda: ("6x5x4x3_-2", torch.randn(6, 5, 4, 3), -2),
            lambda: ("6x5x4x3_-3", torch.randn(6, 5, 4, 3), -3),
            lambda: ("6x5x4x3_-4", torch.randn(6, 5, 4, 3), -4),
            lambda: (
                "3x4_0,int64",
                torch.torch.randint(-10, 10, (3, 4), dtype=torch.int64),
                0,
            ),
        ]
    )
    def test_cumsum(self, _, tensor, dim):
        utils.compare_tracing_methods(
            SimpleCumSumModule(dim), tensor, fusible_ops={"aten::cumsum"}
        )
