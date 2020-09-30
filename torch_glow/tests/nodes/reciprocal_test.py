from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestReciprocal(unittest.TestCase):
    def test_reciprocal(self):
        """Test of the PyTorch reciprocal Node on Glow."""

        def test_f(a):
            return torch.reciprocal(a + a)

        x = torch.randn(4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::reciprocal"})

    def test_inplace_reciprocal(self):
        """Test of the PyTorch inplace reciprocal Node on Glow."""

        def test_f(a):
            b = a + a
            return b.reciprocal_()

        x = torch.randn(4)

        # Expect fuser to out-of-place the operator
        jitVsGlow(test_f, x, expected_fused_ops={"aten::reciprocal"})
