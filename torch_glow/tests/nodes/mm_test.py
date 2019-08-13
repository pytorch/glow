from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_mm_basic():
    """Test of the PyTorch MatMul Node on Glow."""

    def test_mm(a, b, t):
        r = torch.mm(a, b)
        return r.mm(t)

    lhs = torch.randn(2, 3)
    rhs = torch.randn(3, 4)
    t = torch.randn(4, 2)

    jitVsGlow(test_mm, lhs, rhs, t, expected_fused_ops={"aten::mm"})
