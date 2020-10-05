from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestMul(unittest.TestCase):
    def test_mul_basic(self):
        """Basic test of the PyTorch mul Node on Glow."""

        def test_f(a, b):
            c = a.mul(b)
            return c.mul(c)

        x = torch.randn(4)
        y = torch.randn(4)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::mul"})

    def test_mul_broadcast_1(self):
        """Test of the PyTorch mul Node on Glow with broadcasting."""

        def test_f(a, b):
            c = a.mul(b)
            return c.mul(c)

        x = torch.randn(8, 3, 4, 2)
        y = torch.randn(4, 2)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::mul"})

    def test_mul_broadcast_2(self):
        """Test of the PyTorch mul Node on Glow with broadcasting."""

        def test_f(a, b):
            c = a.mul(b)
            return c.mul(c)

        x = torch.randn(8, 3, 4, 2)
        y = torch.randn(1, 2)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::mul"})

    def test_mul_broadcast_3(self):
        """Test of the PyTorch mul Node on Glow with broadcasting."""

        def test_f(a, b):
            c = a.mul(b)
            return c.mul(c)

        x = torch.randn(4, 2)
        y = torch.randn(8, 3, 4, 2)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::mul"})

    def test_mul_float(self):
        """Test of the PyTorch aten::mul Node with a float argument"""

        def test_f(a):
            return (a + a).mul(3.9)

        x = torch.randn(4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::mul"})

    def test_mul_int(self):
        """Test of the PyTorch aten::mul Node with an int argument"""

        def test_f(a):
            return (a + a).mul(20)

        x = torch.randn(4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::mul"})
