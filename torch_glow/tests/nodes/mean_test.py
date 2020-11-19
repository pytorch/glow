from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleMeanModule(torch.nn.Module):
    def __init__(self, dim=None):
        super(SimpleMeanModule, self).__init__()
        self.dim = dim

    def forward(self, a, b):
        if self.dim:
            return torch.mean(a + b, self.dim)
        else:
            return torch.mean(a + b)


class TestMean(unittest.TestCase):
    def test_basic(self):
        """Test of the PyTorch mean Node on Glow."""

        utils.compare_tracing_methods(
            SimpleMeanModule(),
            torch.randn(7),
            torch.randn(7),
            fusible_ops={"aten::mean"},
        )

    def test_with_dims(self):
        """Test of the PyTorch mean node with dims on Glow. """

        utils.compare_tracing_methods(
            SimpleMeanModule((1, 2)),
            torch.randn([1, 2, 3, 4]),
            torch.randn([1, 2, 3, 4]),
            fusible_ops={"aten::mean"},
        )
