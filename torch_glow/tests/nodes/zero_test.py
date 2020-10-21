from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestZero(unittest.TestCase):
    def test_zero_basic(self):
        """Basic test of the PyTorch zero Node on Glow."""

        def test_f(a):
            b = torch.zeros(a.size(), dtype=torch.float)
            return a + b

        x = torch.randn(2, 3, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::zeros"})
