from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests.utils import jitVsGlow


def test_elementwise_max():
    """Test of the PyTorch max Node on Glow."""

    def test_f(a, b):
        c = torch.max(a, b)
        return torch.max(c, c)

    x = torch.randn(4)
    y = torch.randn(4)

    jitVsGlow(test_f, x, y, expected_fused_ops={"aten::max"})
