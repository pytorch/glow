from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestSqrt(unittest.TestCase):
    def test_sqrt_basic(self):
        """Test of the PyTorch sqrt Node on Glow."""

        def test_f(a):
            b = torch.sqrt(a)
            return torch.sqrt(b)

        # Make sure the input is positive and not super close to zero.
        x = torch.rand(4) + 5

        jitVsGlow(test_f, x, expected_fused_ops={"aten::sqrt"})

    def test_sqrt_inplace(self):
        """Test of the PyTorch inplace sqrt Node on Glow."""

        def test_f(a):
            b = a.sqrt_()
            return b.sqrt_()

        # Make sure the input is positive and not super close to zero.
        x = torch.rand(4) + 5

        jitVsGlow(test_f, x, expected_fused_ops={"aten::sqrt_"})
