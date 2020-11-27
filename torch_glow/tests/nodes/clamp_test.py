from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleClampModel(torch.nn.Module):
    def __init__(self, min, max):
        super(SimpleClampModel, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input):
        return torch.clamp(input, self.min, self.max)


class TestClamp(unittest.TestCase):
    def test_clamp(self):
        """Test of the PyTorch clamp Node on Glow."""

        utils.compare_tracing_methods(
            SimpleClampModel(0.0, 6.0), torch.randn(7), fusible_ops={"aten::clamp"}
        )
