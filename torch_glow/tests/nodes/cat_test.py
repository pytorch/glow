from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleCatModule(torch.nn.Module):
    def __init__(self, *dimensions):
        super(SimpleCatModule, self).__init__()
        self.dimensions = dimensions

    def forward(self, a, b, c):
        other = torch.cat((a, b, c), self.dimensions[0])
        for dimension in self.dimensions[1:]:
            other = torch.cat((other, other), dimension)
        return other


class TestCat(utils.TorchGlowTestCase):
    def test_cat_with_empty_tensor(self):
        """Basic test of the PyTorch cat Node on Glow."""

        utils.compare_tracing_methods(
            SimpleCatModule(0, 1, 2),
            torch.empty(0),
            torch.randn(2, 3, 4, 5),
            torch.randn(2, 3, 4, 5),
            fusible_ops={"prim::FusedConcat"},
        )

    def test_cat_basic(self):
        """Basic test of the PyTorch cat Node on Glow."""

        utils.compare_tracing_methods(
            SimpleCatModule(0, 1, 2),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
            fusible_ops={"prim::FusedConcat"},
        )

    def test_cat_neg_dim(self):
        """Test negative dimension index for the PyTorch cat Node on Glow."""

        utils.compare_tracing_methods(
            SimpleCatModule(-3, -2, -1),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
            fusible_ops={"prim::FusedConcat"},
        )

    def test_cat_oob_neg_dim(self):
        """Test out of bounds negative dimension index for the PyTorch cat Node on Glow."""

        with self.assertRaises(IndexError):
            utils.compare_tracing_methods(
                SimpleCatModule(-4, -2, -1),
                torch.randn(2, 3, 4),
                torch.randn(2, 3, 4),
                torch.randn(2, 3, 4),
                fusible_ops={"prim::FusedConcat"},
            )

    def test_cat_with_different_types(self):
        """Test cat between different types that can be cast, which is supported in pytorch."""

        utils.compare_tracing_methods(
            SimpleCatModule(0, 1, 2),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4, dtype=torch.half),
            torch.randn(2, 3, 4, dtype=torch.half),
            fusible_ops={"prim::FusedConcat"},
        )

        utils.compare_tracing_methods(
            SimpleCatModule(0, 1, 2),
            torch.randn(2, 3, 4).to(torch.int),
            torch.randn(2, 3, 4).to(torch.long),
            torch.randn(2, 3, 4).to(torch.long),
            fusible_ops={"prim::FusedConcat"},
        )
