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


class TestFullLike(utils.TorchGlowTestCase):
    def test_empty_like_basic(self):
        """Basic test of the PyTorch empty_like Node on Glow."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.empty_like(a, dtype=torch.float)
                c = torch.zeros_like(a, dtype=torch.float)
                return a + (b * c)

        x = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(TestModule(), x, fusible_ops={"aten::empty_like"})

    def test_empty_like_no_assign_type(self):
        """Basic test of the PyTorch empty_like Node on Glow without assigning type."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.empty_like(a)
                c = torch.zeros_like(a)
                return a + (b * c)

        x = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(TestModule(), x, fusible_ops={"aten::empty_like"})

    def test_empty_like_int(self):
        """Basic test of the PyTorch empty_like Node on Glow with int type."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.empty_like(a, dtype=torch.int)
                c = torch.zeros_like(a, dtype=torch.int)
                return b * c

        x = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(TestModule(), x, fusible_ops={"aten::empty_like"})

    def test_full_like_basic(self):
        """Basic test of the PyTorch full_like Node on Glow."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.full_like(a, fill_value=3.1415, dtype=torch.float)
                return a + b

        x = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(TestModule(), x, fusible_ops={"aten::full_like"})

    def test_full_like_no_assign_type(self):
        """Basic test of the PyTorch full_like Node on Glow without assigning type."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.full_like(a, fill_value=3.1415)
                return a + b

        x = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(TestModule(), x, fusible_ops={"aten::full_like"})

    def test_full_like_int(self):
        """Basic test of the PyTorch full_like Node on Glow with int type."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.full_like(a, fill_value=4, dtype=torch.int)
                c = torch.full_like(a, fill_value=5, dtype=torch.int)
                return b + c

        x = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(TestModule(), x, fusible_ops={"aten::full_like"})

    def test_zeros_like_basic(self):
        """Basic test of the PyTorch zeros_like Node on Glow."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.zeros_like(a, dtype=torch.float)
                return a + b

        x = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(TestModule(), x, fusible_ops={"aten::zeros_like"})

    def test_zeros_like_no_assign_type(self):
        """Basic test of the PyTorch zeros_like Node on Glow without assign type."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.zeros_like(a)
                return a + b

        x = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(TestModule(), x, fusible_ops={"aten::zeros_like"})

    def test_zeros_like_int(self):
        """Basic test of the PyTorch zeros_like Node on Glow with int type."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.zeros_like(a, dtype=torch.int)
                c = torch.zeros_like(b)
                return b + c

        x = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(TestModule(), x, fusible_ops={"aten::zeros_like"})

    def test_ones_like_basic(self):
        """Basic test of the PyTorch ones_like Node on Glow."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.ones_like(a, dtype=torch.float)
                return a + b

        x = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(TestModule(), x, fusible_ops={"aten::ones_like"})

    def test_ones_like_no_assign_type(self):
        """Basic test of the PyTorch ones_like Node on Glow without assign type."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.ones_like(a)
                return a + b

        x = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(TestModule(), x, fusible_ops={"aten::ones_like"})

    def test_ones_like_int(self):
        """Basic test of the PyTorch ones_like Node on Glow with int type."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.ones_like(a, dtype=torch.int)
                c = torch.ones_like(b, dtype=torch.int)
                return b + c

        x = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(TestModule(), x, fusible_ops={"aten::ones_like"})
