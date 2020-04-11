from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow
import unittest


class TestRsub(unittest.TestCase):
    def test_rsub_basic(self):
        """Basic test of the PyTorch rsub Node on Glow."""

        def test_f(a, b):
            c = torch.rsub(a, b)
            return torch.rsub(c, c)

        x = torch.randn(4)
        y = torch.randn(4)

        jitVsGlow(test_f, x, y,  expected_fused_ops={"aten::rsub"})

    def test_rsub_broadcast_1(self):
        """Test of the PyTorch rsub Node on Glow with broadcasting."""

        def test_f(a, b):
            c = torch.rsub(a, b)
            return torch.rsub(c, c)

        x = torch.randn(8, 3, 4, 2)
        y = torch.randn(4, 2)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::rsub"})

    def test_rsub_broadcast_2(self):
        """Test of the PyTorch rsub Node on Glow with broadcasting."""

        def test_f(a, b):
            c = torch.rsub(a, b)
            return torch.rsub(c, c)

        x = torch.randn(8, 3, 4, 2)
        y = torch.randn(1, 2)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::rsub"})

    def test_rsub_broadcast_3(self):
        """Test of the PyTorch rsub Node on Glow with broadcasting."""

        def test_f(a, b):
            c = torch.rsub(a, b)
            return torch.rsub(c, c)

        x = torch.randn(4, 2)
        y = torch.randn(8, 3, 4, 2)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::rsub"})

    def test_rsub_float(self):
        """Test of the PyTorch aten::rsub Node with a float argument"""

        def test_f(a):
            return torch.rsub((a * a), 3.9)

        x = torch.randn(4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::rsub"})

    def test_rsub_int(self):
        """Test of the PyTorch aten::rsub Node with an int argument"""

        def test_f(a):
            return torch.rsub((a * a), 20)

        x = torch.randn(4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::rsub"})
