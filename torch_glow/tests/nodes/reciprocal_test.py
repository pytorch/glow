from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from tests.utils import jitVsGlow


def test_reciprocal():
    """Test of the PyTorch reciprocal Node on Glow."""

    def reciprocal(a):
        return torch.reciprocal(a + a)

    x = torch.randn(4)

    jitVsGlow(reciprocal, x)


def test_inplace_reciprocal():
    """Test of the PyTorch inplace reciprocal Node on Glow."""

    def reciprocal_inplace(a):
        b = a + a
        return b.reciprocal_()

    x = torch.randn(4)

    jitVsGlow(reciprocal_inplace, x)
