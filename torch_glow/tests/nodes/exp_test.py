from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleExpModule(torch.nn.Module):
    def forward(self, input):
        other = torch.exp(input)
        return torch.exp(other)


class TestExp(unittest.TestCase):
    def test_exp_basic(self):
        """Test of the PyTorch exp Node on Glow."""

        utils.compare_tracing_methods(
            SimpleExpModule(), torch.randn(4), fusible_ops={"aten::exp"}
        )
