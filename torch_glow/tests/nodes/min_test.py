from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_elementwise_min():
    """Test of the PyTorch min Node on Glow."""

    def test_f(a, b):
        return torch.min(a + a, b + b)

    jitVsGlow(test_f, torch.randn(7), torch.randn(7))
