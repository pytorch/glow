from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_clamp():
    """Test of the PyTorch clamp Node on Glow."""

    def test_f(x):
        return torch.clamp(x, 0.0, 6.0)

    jitVsGlow(test_f, torch.randn(7), expected_fused_ops={"aten::clamp"})
