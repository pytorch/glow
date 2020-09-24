from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestCeil(unittest.TestCase):
    def test_ceil(self):
        """Basic test of the PyTorch Ceil Node on Glow."""

        def test_f(a, b):
            c = a + b
            d = torch.ceil(c)
            return d

        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::ceil"})
