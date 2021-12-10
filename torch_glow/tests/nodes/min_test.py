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


class SimpleMinModule(torch.nn.Module):
    def __init__(self):
        super(SimpleMinModule, self).__init__()

    def forward(self, a, b):
        return torch.min(a + a, b + b)


class UnaryMinModule(torch.nn.Module):
    def __init__(self):
        super(UnaryMinModule, self).__init__()

    def forward(self, a):
        return torch.min(a + a)


class TestMin(utils.TorchGlowTestCase):
    def test_elementwise_min(self):
        """Test of the PyTorch min Node on Glow."""

        utils.compare_tracing_methods(
            SimpleMinModule(), torch.randn(7), torch.randn(7), fusible_ops={"aten::min"}
        )

    def test_elementwise_min_broadcast(self):
        """Test of the PyTorch min Node with broadcast on Glow."""

        utils.compare_tracing_methods(
            SimpleMinModule(),
            torch.randn(2, 7),
            torch.randn(7),
            fusible_ops={"aten::min"},
        )

    def test_unary_min(self):
        """Test of the PyTorch unary min Node on Glow."""

        utils.compare_tracing_methods(
            UnaryMinModule(),
            torch.randint(
                20,
                (
                    10,
                    10,
                ),
                dtype=torch.int32,
            ),
            fusible_ops={"aten::min"},
        )
