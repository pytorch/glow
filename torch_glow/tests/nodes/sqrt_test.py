from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from tests.utils import jitVsGlow


def test_sqrt_basic():
    """Test of the PyTorch sqrt Node on Glow."""

    def sqrt_basic(a):
        b = torch.sqrt(a)
        return torch.sqrt(b)

    # Make sure the input is positive and not super close to zero.
    x = torch.rand(4) + 5

    jitVsGlow(sqrt_basic, x)


def test_sqrt_inplace():
    """Test of the PyTorch inplace sqrt Node on Glow."""

    def sqrt_inplace(a):
        b = a.sqrt_()
        return b.sqrt_()

    # Make sure the input is positive and not super close to zero.
    x = torch.rand(4) + 5

    jitVsGlow(sqrt_inplace, x)
