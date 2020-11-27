from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleSqrtModel(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleSqrtModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        if self.inplace:
            other = tensor.sqrt_()
            return other.sqrt_()
        else:
            tensor = torch.sqrt(tensor)
            return torch.sqrt(tensor)


class TestSqrt(unittest.TestCase):
    def test_sqrt_basic(self):
        """Test of the PyTorch sqrt Node on Glow."""

        # Make sure the input is positive and not super close to zero.
        utils.compare_tracing_methods(SimpleSqrtModel(), torch.rand(4) + 5)

    def test_sqrt_inplace(self):
        """Test of the PyTorch inplace sqrt Node on Glow."""

        # Make sure the input is positive and not super close to zero.
        utils.compare_tracing_methods(SimpleSqrtModel(inplace=True), torch.rand(4) + 5)
