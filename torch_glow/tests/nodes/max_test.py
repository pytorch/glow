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


class SimpleMaxModule(torch.nn.Module):
    def __init__(self):
        super(SimpleMaxModule, self).__init__()

    def forward(self, a, b):
        return torch.max(a + a, b + b)


class UnaryMaxModule(torch.nn.Module):
    def __init__(self):
        super(UnaryMaxModule, self).__init__()

    def forward(self, a):
        return torch.max(a + a)


class ReduceMaxModule(torch.nn.Module):
    def __init__(self, dim, keep_dim):
        super(ReduceMaxModule, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, a):
        values, index = torch.max(a + a, self.dim, self.keep_dim)
        return torch.stack((values, index))


class TestMax(utils.TorchGlowTestCase):
    def test_elementwise_max(self):
        """Test of the PyTorch max Node on Glow."""

        utils.compare_tracing_methods(
            SimpleMaxModule(), torch.randn(4), torch.randn(4), fusible_ops={"aten::max"}
        )

    def test_elementwise_max_broadcast(self):
        """Test of the PyTorch max Node with broadcast on Glow."""

        utils.compare_tracing_methods(
            SimpleMaxModule(),
            torch.randn(2, 4),
            torch.randn(4),
            fusible_ops={"aten::max"},
        )

    def test_unary_max(self):
        """Test of the PyTorch unary max Node on Glow."""

        utils.compare_tracing_methods(
            UnaryMaxModule(),
            torch.randint(
                50,
                (
                    10,
                    10,
                ),
                dtype=torch.int,
            ),
            fusible_ops={"aten::max"},
        )

    def test_reduce_max(self):
        """Test of the PyTorch max Node reducing on a specified dim."""

        utils.compare_tracing_methods(
            ReduceMaxModule(2, False),
            torch.randn(3, 4, 5),
            fusible_ops={"aten::max"},
        )

    def test_reduce_max_keep_dim(self):
        """Test of the PyTorch max Node reducing on a specified dim and keeping dim."""

        utils.compare_tracing_methods(
            ReduceMaxModule(2, True),
            torch.randn(3, 4, 5),
            fusible_ops={"aten::max"},
        )
