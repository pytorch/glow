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


class SimpleClampModel(torch.nn.Module):
    def __init__(self, min, max):
        super(SimpleClampModel, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input):
        return torch.clamp(input, self.min, self.max)


class TestClamp(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", 0.0, 0.8, torch.float),
            lambda: ("no_min", None, 0.8, torch.float),
            lambda: ("no_max", 0.0, None, torch.float),
            lambda: ("int_basic", 4, 8, torch.int32),
        ]
    )
    def test_clamp(self, _, min, max, dtype):
        """Test of the PyTorch clamp Node on Glow."""
        a = torch.randn(2, 7)
        if dtype == torch.int32:
            a = torch.randint(max * 2, (2, 7))

        utils.compare_tracing_methods(
            SimpleClampModel(min, max), a, fusible_ops={"aten::clamp"}
        )
