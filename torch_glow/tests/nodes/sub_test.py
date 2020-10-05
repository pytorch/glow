from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestSub(unittest.TestCase):
    def test_sub_basic(self):
        """Basic test of the PyTorch sub Node on Glow."""

        def test_f(a, b):
            c = a.sub(b)
            return c.sub(c)

        x = torch.randn(4)
        y = torch.randn(4)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::sub"})

    def test_sub_broadcast_1(self):
        """Test of the PyTorch sub Node on Glow with broadcasting."""

        def test_f(a, b):
            c = a.sub(b)
            return c.sub(c)

        x = torch.randn(8, 3, 4, 2)
        y = torch.randn(4, 2)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::sub"})

    def test_sub_broadcast_2(self):
        """Test of the PyTorch sub Node on Glow with broadcasting."""

        def test_f(a, b):
            c = a.sub(b)
            return c.sub(c)

        x = torch.randn(8, 3, 4, 2)
        y = torch.randn(1, 2)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::sub"})

    def test_sub_broadcast_3(self):
        """Test of the PyTorch sub Node on Glow with broadcasting."""

        def test_f(a, b):
            c = a.sub(b)
            return c.sub(c)

        x = torch.randn(4, 2)
        y = torch.randn(8, 3, 4, 2)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::sub"})

    def test_sub_float(self):
        """Test of the PyTorch aten::sub Node with a float argument"""

        def test_f(a):
            return (a * a).sub(3.9)

        x = torch.randn(4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::sub"})

    def test_sub_int(self):
        """Test of the PyTorch aten::sub Node with an int argument"""

        def test_f(a):
            return (a * a).sub(20)

        x = torch.randn(4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::sub"})
