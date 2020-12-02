from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleClampModel(torch.nn.Module):
    def __init__(self, min, max, inplace=False):
        super(SimpleClampModel, self).__init__()
        self.min = min
        self.max = max
        self.inplace = inplace

    def forward(self, input):
        if self.inplace:
            return input.clamp_(min=self.min, max=self.max)
        else:
            return torch.clamp(input, self.min, self.max)


class TestClamp(unittest.TestCase):
    def test_clamp(self):
        """Test of the PyTorch clamp Node on Glow."""

        utils.compare_tracing_methods(
            SimpleClampModel(0.0, 6.0), torch.randn(7), fusible_ops={"aten::clamp"}
        )

    def test_clamp_int(self):
        """Test of the PyTorch clamp Node with integer inputs on Glow."""

        utils.compare_tracing_methods(
            SimpleClampModel(0, 6), torch.arange(7), fusible_ops={"aten::clamp"}
        )

    def test_clamp_inplace(self):
        """Test of the PyTorch clamp_ (inplace) Node on Glow."""

        utils.compare_tracing_methods(
            SimpleClampModel(3.0, 6.0, inplace=True),
            torch.randn(7),
            fusible_ops={"aten::clamp_"},
        )

    def test_clamp_int_inplace(self):
        """Test of the PyTorch clamp_ (inplace) Node with integer inputs on Glow."""

        utils.compare_tracing_methods(
            SimpleClampModel(3, 6, inplace=True),
            torch.arange(7),
            fusible_ops={"aten::clamp_"},
        )
