from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleMaxModule(torch.nn.Module):
    def __init__(self):
        super(SimpleMaxModule, self).__init__()

    def forward(self, a, b):
        return torch.max(a + a, b + b)


class TestMax(unittest.TestCase):
    def test_elementwise_max(self):
        """Test of the PyTorch max Node on Glow."""

        utils.compare_tracing_methods(
            SimpleMaxModule(), torch.randn(4), torch.randn(4), fusible_ops={"aten::max"}
        )
