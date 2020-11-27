from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleCatModule(torch.nn.Module):
    def __init__(self, *dimensions):
        super(SimpleCatModule, self).__init__()
        self.dimensions = dimensions

    def forward(self, a, b):
        other = torch.cat((a, b), self.dimensions[0])
        for dimension in self.dimensions[1:]:
            other = torch.cat((other, other), dimension)
        return other


class TestCat(unittest.TestCase):
    def test_cat_basic(self):
        """Basic test of the PyTorch cat Node on Glow."""

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(
            SimpleCatModule(0, 1, 2),
            x,
            y,
            fusible_ops={"prim::FusedConcat"},
        )

    def test_cat_neg_dim(self):
        """Test negative dimension index for the PyTorch cat Node on Glow."""

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(
            SimpleCatModule(-3, -2, -1),
            x,
            y,
            fusible_ops={"prim::FusedConcat"},
        )

    def test_cat_oob_neg_dim(self):
        """Test out of bounds negative dimension index for the PyTorch cat Node on Glow."""

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        with self.assertRaises(IndexError):
            utils.compare_tracing_methods(
                SimpleCatModule(-4, -2, -1),
                x,
                y,
                fusible_ops={"prim::FusedConcat"},
            )

    def test_cat_with_different_types(self):
        """Test cat between different types that can be cast, which is supported in pytorch."""

        utils.compare_tracing_methods(
            SimpleCatModule(0, 1, 2),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4, dtype=torch.half),
            fusible_ops={"prim::FusedConcat"},
        )

        utils.compare_tracing_methods(
            SimpleCatModule(0, 1, 2),
            torch.randn(2, 3, 4).to(torch.int),
            torch.randn(2, 3, 4).to(torch.long),
            fusible_ops={"prim::FusedConcat"},
        )
