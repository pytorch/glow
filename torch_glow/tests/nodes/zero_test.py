from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class TestZero(unittest.TestCase):
    def test_zero_basic(self):
        """Basic test of the PyTorch zero Node on Glow."""

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.zeros(a.size(), dtype=torch.float)
                return a + b

        x = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(TestModule(), x, fusible_ops={"aten::zeros"})
