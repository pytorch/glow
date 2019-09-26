from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_mm_basic():
    """Test of the PyTorch mm Node on Glow."""

    def test_f(a, b, t):
        r = torch.mm(a, b)
        return r.mm(t)

    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    t = torch.randn(4, 2)

    jitVsGlow(test_f, x, y, t, expected_fused_ops={"aten::mm"})
