from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestBmm(unittest.TestCase):
    def test_bmm(self):
        """Basic test of the PyTorch bmm Node on Glow."""

        def test_f(a, b):
            return (a + a).bmm(b)

        x = torch.randn(6, 4, 10)
        y = torch.randn(6, 10, 2)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::bmm"})
