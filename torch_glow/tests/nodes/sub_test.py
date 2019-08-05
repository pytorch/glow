from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_sub_basic():
    """Basic test of the PyTorch sub Node on Glow."""

    def test_f(a, b):
        c = a.sub(b)
        return c.sub(c)

    x = torch.randn(4)
    y = torch.randn(4)

    jitVsGlow(test_f, x, y)


def test_sub_broadcast_1():
    """Test of the PyTorch sub Node on Glow with broadcasting."""

    def test_f(a, b):
        c = a.sub(b)
        return c.sub(c)

    x = torch.randn(8, 3, 4, 2)
    y = torch.randn(4, 2)

    jitVsGlow(test_f, x, y)


def test_sub_broadcast_2():
    """Test of the PyTorch sub Node on Glow with broadcasting."""

    def test_f(a, b):
        c = a.sub(b)
        return c.sub(c)

    x = torch.randn(8, 3, 4, 2)
    y = torch.randn(1, 2)

    jitVsGlow(test_f, x, y)


def test_sub_broadcast_3():
    """Test of the PyTorch sub Node on Glow with broadcasting."""

    def test_f(a, b):
        c = a.sub(b)
        return c.sub(c)

    x = torch.randn(4, 2)
    y = torch.randn(8, 3, 4, 2)

    jitVsGlow(test_f, x, y)
