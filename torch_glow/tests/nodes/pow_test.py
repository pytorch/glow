from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimplePowModule(torch.nn.Module):
    def __init__(self, power):
        super(SimplePowModule, self).__init__()
        self.power = power

    def forward(self, tensor):
        return torch.pow(tensor, self.power)


class TestPow(unittest.TestCase):
    @parameterized.expand(
        [
            ("float", 2.2),
            ("tensor_basic", torch.randn(4) + 2),
            ("tensor_size[]", torch.tensor(2.2)),
            ("tensor_broadcast", torch.randn(1) + 2),
        ]
    )
    def test_pow_basic(self, _, power):
        """Test of the PyTorch pow Node on Glow."""

        utils.compare_tracing_methods(
            SimplePowModule(power), torch.rand(4) + 5, fusible_ops={"aten::pow"}
        )
