from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow

from tests.utils import jitVsGlow

# Basic test of the PyTorch div Node on Glow.


def test_div_basic():
    def div_basic(a, b):
        c = a.div(b)
        return c.div(c)

    x = torch.randn(4)
    y = torch.randn(4)

    jitVsGlow(div_basic, x, y)

# Test of the PyTorch div Node on Glow with broadcasting.


def test_div_broadcast_1():

    def test_f(a, b):
        c = a.div(b)
        return c.div(c)

    x = torch.randn(8, 3, 4, 2)
    y = torch.randn(4, 2)

    jitVsGlow(test_f, x, y)

# Test of the PyTorch div Node on Glow with broadcasting.


def test_div_broadcast_2():

    def test_f(a, b):
        c = a.div(b)
        return c.div(c)

    x = torch.randn(8, 3, 4, 2)
    y = torch.randn(1, 2)

    jitVsGlow(test_f, x, y)

    # Test of the PyTorch div Node on Glow with broadcasting.


def test_div_broadcast_3():

    def test_f(a, b):
        c = a.div(b)
        return c.div(c)

    x = torch.randn(4, 2)
    y = torch.randn(8, 3, 4, 2)

    jitVsGlow(test_f, x, y)
