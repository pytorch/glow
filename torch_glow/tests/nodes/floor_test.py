from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleFloorModule(torch.nn.Module):
    def forward(self, a, b):
        c = a + b
        return torch.floor(c)


class TestFloor(unittest.TestCase):
    def test_floor(self):
        """Basic test of the PyTorch floor Node on Glow."""

        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        utils.compare_tracing_methods(
            SimpleFloorModule(), x, y, fusible_ops={"aten::floor"}
        )
