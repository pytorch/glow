from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_add_basic():
    """Basic test of the PyTorch add Node on Glow."""

    def test_f(a, b):
        c = a.add(b)
        return c.add(c)

    x = torch.randn(4)
    y = torch.randn(4)

    jitVsGlow(test_f, x, y, expected_fused_ops={"aten::add"})


def test_add_inplace():
    """Test of the PyTorch add_ Node on Glow."""

    def test_f(a, b):
        c = a.add_(b)
        return c.add_(c)

    x = torch.randn(4)
    y = torch.randn(4)

    jitVsGlow(test_f, x, y, expected_fused_ops={"aten::add_"})


def test_add_broadcast_1():
    """Test of the PyTorch add Node on Glow with broadcasting."""

    def test_f(a, b):
        c = a.add(b)
        return c.add(c)

    x = torch.randn(8, 3, 4, 2)
    y = torch.randn(4, 2)

    jitVsGlow(test_f, x, y, expected_fused_ops={"aten::add"})


def test_add_broadcast_2():
    """Test of the PyTorch add Node on Glow with broadcasting."""

    def test_f(a, b):
        c = a.add(b)
        return c.add(c)

    x = torch.randn(8, 3, 4, 2)
    y = torch.randn(1, 2)

    jitVsGlow(test_f, x, y, expected_fused_ops={"aten::add"})


def test_add_broadcast_3():
    """Test of the PyTorch add Node on Glow with broadcasting."""

    def test_f(a, b):
        c = a.add(b)
        return c.add(c)

    x = torch.randn(4, 2)
    y = torch.randn(8, 3, 4, 2)

    jitVsGlow(test_f, x, y, expected_fused_ops={"aten::add"})


def test_add_float():
    """Test of the PyTorch aten::add Node with a float argument"""

    def test_f(a):
        return (a*a).add(3.9)

    x = torch.randn(4)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::add"})


def test_add_int():
    """Test of the PyTorch aten::add Node with an int argument"""

    def test_f(a):
        return (a*a).add(20)

    x = torch.randn(4)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::add"})
