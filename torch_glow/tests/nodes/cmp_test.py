from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestCmp(unittest.TestCase):
    def test_equal(self):
        """Basic test of the PyTorch Equal Node on Glow."""

        def test_f(a, b):
            return torch.eq(a, b + 0.1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::eq"})

    def test_equal_bcast(self):
        """Basic test of the PyTorch Equal Node (broadcast) on Glow."""

        def test_f(a, b):
            return torch.eq(a, b + 0.1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(4, 5)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::eq"})

    def test_not_equal(self):
        """Basic test of the PyTorch Not equal Node on Glow."""

        def test_f(a, b):
            return torch.ne(a, b + 0.1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::ne"})

    def test_not_equal_bcast(self):
        """Basic test of the PyTorch Not equal (broadcast) Node on Glow."""

        def test_f(a, b):
            return torch.ne(a, b + 0.1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(4, 5)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::ne"})

    def test_less_than(self):
        """Basic test of the PyTorch Less than Node on Glow."""

        def test_f(a, b):
            return torch.lt(a, b + 0.1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::lt"})

    def test_less_equal(self):
        """Basic test of the PyTorch less equal Node on Glow."""

        def test_f(a, b):
            return torch.le(a, b + 0.1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::le"})

    def test_less_than_bcast(self):
        """Basic test of the PyTorch Less than (broadcast) Node on Glow."""

        def test_f(a, b):
            return torch.lt(a, b + 0.1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(4, 5)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::lt"})

    def test_less_equal_bcast(self):
        """Basic test of the PyTorch less equal (broadcast) Node on Glow."""

        def test_f(a, b):
            return torch.le(a, b + 0.1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(4, 5)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::le"})

    def test_greater_than(self):
        """Basic test of the PyTorch Greater than Node on Glow."""

        def test_f(a, b):
            return torch.gt(a, b + 0.1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::gt"})

    def test_greater_equal(self):
        """Basic test of the PyTorch Greater Equal Node on Glow."""

        def test_f(a, b):
            return torch.ge(a, b + 1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::ge"})

    def test_greater_than_bcast(self):
        """Basic test of the PyTorch Greater than (broadcast) Node on Glow."""

        def test_f(a, b):
            return torch.gt(a, b + 0.1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(4, 5)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::gt"})

    def test_greater_equal_bcast(self):
        """Basic test of the PyTorch Greater Equal (broadcast) Node on Glow."""

        def test_f(a, b):
            return torch.ge(a, b + 1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(4, 5)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::ge"})
