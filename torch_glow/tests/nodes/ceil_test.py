from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleCeilModule(torch.nn.Module):
    def forward(self, a, b):
        c = a + b
        return torch.ceil(c)


class TestCeil(unittest.TestCase):
    def test_ceil(self):
        """Basic test of the PyTorch Ceil Node on Glow."""

        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        utils.compare_tracing_methods(
            SimpleCeilModule(), x, y, fusible_ops={"aten::ceil"}
        )
