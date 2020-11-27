from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleAbsModule(torch.nn.Module):
    def __init__(self):
        super(SimpleAbsModule, self).__init__()

    def forward(self, a):
        return torch.abs(a + a)


class TestAbs(unittest.TestCase):
    def test_abs_basic(self):
        """Basic test of the PyTorch Abs Node on Glow."""

        x = torch.randn(10)
        utils.compare_tracing_methods(
            SimpleAbsModule(),
            x,
            fusible_ops={"aten::abs"},
        )

    def test_abs_3d(self):
        """Test multidimensional tensor for the PyTorch Abs Node on Glow."""

        x = torch.randn(2, 3, 5)
        utils.compare_tracing_methods(
            SimpleAbsModule(),
            x,
            fusible_ops={"aten::abs"},
        )
