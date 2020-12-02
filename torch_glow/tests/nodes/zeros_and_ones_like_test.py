from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleLikeModule(torch.nn.Module):
    def __init__(self, splat_val):
        super(SimpleLikeModule, self).__init__()
        self.splat_val = splat_val

    def forward(self, tensor):
        if self.splat_val == 0:
            return torch.zeros_like(tensor + tensor)
        else:
            return torch.ones_like(tensor + tensor)


class TestZerosLike(unittest.TestCase):
    def test_zeros_like_basic(self):
        """Test of the PyTorch zeros_like Node on Glow."""

        utils.compare_tracing_methods(
            SimpleLikeModule(splat_val=0),
            torch.rand(4, 2, 3),
            fusible_ops={"aten::zeros_like"},
        )


class TestOnesLike(unittest.TestCase):
    def test_ones_like_basic(self):
        """Test of the PyTorch ones_like Node on Glow."""

        utils.compare_tracing_methods(
            SimpleLikeModule(splat_val=1),
            torch.rand(4, 2, 3),
            fusible_ops={"aten::ones_like"},
        )