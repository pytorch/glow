from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow

from tests.utils import jitVsGlow

# Test of the PyTorch exp Node on Glow.


def test_exp_basic():
    def exp_basic(a):
        b = torch.exp(a)
        return torch.exp(b)

    x = torch.randn(4)

    jitVsGlow(exp_basic, x)
