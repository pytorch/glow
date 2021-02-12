from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class TestFullLike(unittest.TestCase):
    def test_empty_like_basic(self):
        """Basic test of the PyTorch empty_like Node on Glow."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.empty_like(a, dtype=torch.float)
                c = torch.zeros_like(a, dtype=torch.float)
                return a + (b * c)

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

    def test_zeros_like_basic(self):
        """Basic test of the PyTorch zeros_like Node on Glow."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.zeros_like(a, dtype=torch.float)
                return a + b

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
