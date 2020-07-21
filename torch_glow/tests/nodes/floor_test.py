from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow
import unittest


class TestFloor(unittest.TestCase):
    def test_floor(self):
        """Basic test of the PyTorch floor Node on Glow."""

        def test_f(a, b):
            c = a + b
            d = torch.floor(c)
            return d

        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        jitVsGlow(test_f,
                  x,
                  y,
                  expected_fused_ops={"aten::floor"})
