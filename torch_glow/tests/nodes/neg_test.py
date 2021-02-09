from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleNegModule(torch.nn.Module):
    def __init__(self):
        super(SimpleNegModule, self).__init__()

    def forward(self, a):
        return torch.neg(a + a)


class TestNeg(unittest.TestCase):
    def test_neg_basic(self):
        """Basic test of the PyTorch Neg Node on Glow."""

        x = torch.randn(10)
        utils.compare_tracing_methods(
            SimpleNegModule(),
            x,
            fusible_ops={"aten::neg"},
        )

    def test_neg_3d(self):
        """Test multidimensional tensor for the PyTorch Neg Node on Glow."""

        x = torch.randn(2, 3, 5)
        utils.compare_tracing_methods(
            SimpleNegModule(),
            x,
            fusible_ops={"aten::neg"},
        )
