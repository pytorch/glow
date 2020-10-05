from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestTo(unittest.TestCase):
    def test_to_basic(self):
        """Test of the PyTorch to Node on Glow."""

        def test_f(a):
            b = a.to(torch.int)
            c = b.to(torch.float)
            return c

        x = torch.randn(1, 2, 3, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::to"})

    def test_to_int(self):
        """Test of the PyTorch to Node on Glow with int output."""

        def test_f(a):
            ai = a.to(torch.int)
            return ai

        x = torch.randn(1, 2, 3, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::to"})

    def test_to_float(self):
        """Test of the PyTorch to Node on Glow with float output."""

        def test_f(a):
            ai = a.to(torch.float)
            return ai

        x = torch.randint(100, (1, 2, 3, 4))

        jitVsGlow(test_f, x, expected_fused_ops={"aten::to"})
