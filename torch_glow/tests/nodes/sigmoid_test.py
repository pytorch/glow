from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow

from tests.utils import jitVsGlow

# Basic test of the PyTorch sigmoid Node on Glow


def test_sigmoid_basic():
    def sigmoid_basic(a):
        c = a.sigmoid()
        return c.sigmoid()

    x = torch.randn(6)

    jitVsGlow(sigmoid_basic, x)

def test_sigmoid_inplace():
    def sigmoid_inplace(a):
        c = a.sigmoid_()
        return c.sigmoid_()

    x = torch.randn(6)

    jitVsGlow(sigmoid_inplace, x)
