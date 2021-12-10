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


class SimplePowModule(torch.nn.Module):
    def __init__(self, power):
        super(SimplePowModule, self).__init__()
        self.power = power

    def forward(self, tensor):
        return torch.pow(tensor, self.power)


class TestPow(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("float", 2.2),
            lambda: ("tensor_basic", torch.randn(4) + 2),
            lambda: ("tensor_size[]", torch.tensor(2.2)),
            lambda: ("tensor_broadcast", torch.randn(1) + 2),
        ]
    )
    def test_pow_basic(self, _, power):
        """Test of the PyTorch pow Node on Glow."""

        utils.compare_tracing_methods(
            SimplePowModule(power), torch.rand(4) + 5, fusible_ops={"aten::pow"}
        )
