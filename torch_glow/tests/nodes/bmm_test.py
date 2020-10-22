from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleBmmModule(torch.nn.Module):
    def forward(self, a, b):
        return (a + a).bmm(b)


class TestBmm(unittest.TestCase):
    def test_bmm(self):
        """Basic test of the PyTorch bmm Node on Glow."""

        x = torch.randn(6, 4, 10)
        y = torch.randn(6, 10, 2)

        utils.compare_tracing_methods(
            SimpleBmmModule(), x, y, fusible_ops={"aten::bmm"}
        )
