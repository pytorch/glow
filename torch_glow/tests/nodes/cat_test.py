from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestCat(unittest.TestCase):
    def test_cat_basic(self):
        """Basic test of the PyTorch cat Node on Glow."""

        def test_f(a, b):
            c = torch.cat((a, b), 0)
            d = torch.cat((c, c), 1)
            return torch.cat((d, d), 2)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        jitVsGlow(test_f, x, y, expected_fused_ops={"prim::FusedConcat"})

    def test_cat_neg_dim(self):
        """Test negative dimension index for the PyTorch cat Node on Glow."""

        def test_f(a, b):
            c = torch.cat((a, b), -3)
            d = torch.cat((c, c), -2)
            return torch.cat((d, d), -1)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        jitVsGlow(test_f, x, y, expected_fused_ops={"prim::FusedConcat"})

    def test_cat_oob_neg_dim(self):
        """Test out of bounds negative dimension index for the PyTorch cat Node on Glow."""

        def test_f(a, b):
            c = torch.cat((a, b), -4)
            d = torch.cat((c, c), -2)
            return torch.cat((d, d), -1)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        with self.assertRaises(IndexError):
            jitVsGlow(test_f, x, y, expected_fused_ops={"prim::FusedConcat"})
