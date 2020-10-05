from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestTypeAs(unittest.TestCase):
    def test_typeas_basic(self):
        """Basic test of the PyTorch type_as Node on Glow (float to int32)."""

        def test_f(a, b):
            c = a.type_as(b)
            return c + c

        x = torch.randn(4)
        y = torch.zeros(4, dtype=torch.int32)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::type_as"})

    def test_typeas_basic2(self):
        """Basic test of the PyTorch type_as Node on Glow (int32 to float)."""

        def test_f(a, b):
            c = a.type_as(b)
            return c + c

        x = torch.randn(4).to(dtype=torch.int32)
        y = torch.randn(4)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::type_as"})

    def test_typeas_bool(self):
        """Test of the PyTorch type_as Node on Glow converting bool to float."""

        def test_f(a, b):
            c = a.type_as(b)
            return c + c

        x = torch.randn(4).to(dtype=torch.bool)
        y = torch.randn(4)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::type_as"})

    def test_typeas_self(self):
        """Test of the PyTorch mul Node on Glow doing empty convert (float to float)."""

        def test_f(a, b):
            a = a + a
            c = a.type_as(b)
            return c + c

        x = torch.randn(4)
        y = x

        jitVsGlow(test_f, x, y, expected_fused_ops={})

    def test_typeas_self_f2f2(self):
        """Test of the PyTorch type_as Node on Glow float to float."""

        def test_f(a, b):
            a = a + a
            c = a.type_as(b)
            return c + c

        x = torch.randn(4, 2)
        y = torch.randn(8, 3, 4, 2)

        jitVsGlow(test_f, x, y, expected_fused_ops={})

    def test_typeas_self_f2i2(self):
        """Test of the PyTorch type_as Node on Glow with float to int32"""

        def test_f(a, b):
            a = a + a
            c = a.type_as(b)
            return c + c

        x = torch.randn(4, 2)
        y = torch.randn(8, 3, 4, 2).to(dtype=torch.int32)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::type_as"})
