from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimplePowModule(torch.nn.Module):
    def __init__(self, *powers):
        super(SimplePowModule, self).__init__()
        self.powers = powers

    def forward(self, tensor):
        for power in self.powers:
            tensor = torch.pow(tensor, power)
        return tensor


class TestPow(unittest.TestCase):
    def test_pow_basic(self):
        """Test of the PyTorch pow Node on Glow."""

        utils.compare_tracing_methods(
            SimplePowModule(2.3, 3.4), torch.rand(4) + 5, fusible_ops={"aten::pow"}
        )
