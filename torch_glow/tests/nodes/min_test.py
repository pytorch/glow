from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleMinModule(torch.nn.Module):
    def __init__(self):
        super(SimpleMinModule, self).__init__()

    def forward(self, a, b):
        return torch.min(a + a, b + b)


class TestMin(unittest.TestCase):
    def test_elementwise_min(self):
        """Test of the PyTorch min Node on Glow."""

        utils.compare_tracing_methods(
            SimpleMinModule(), torch.randn(7), torch.randn(7), fusible_ops={"aten::min"}
        )
