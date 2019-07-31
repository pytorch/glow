from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests.utils import jitVsGlow


def test_reciprocal():
    """Test of the PyTorch reciprocal Node on Glow."""

    def test_f(a):
        return torch.reciprocal(a + a)

    x = torch.randn(4)

    jitVsGlow(test_f, x)


def test_inplace_reciprocal():
    """Test of the PyTorch inplace reciprocal Node on Glow."""

    def test_f(a):
        b = a + a
        return b.reciprocal_()

    x = torch.randn(4)

    jitVsGlow(test_f, x)
