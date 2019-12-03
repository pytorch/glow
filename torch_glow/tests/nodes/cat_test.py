from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_cat_basic():
    """Basic test of the PyTorch cat Node on Glow."""

    def test_f(a, b):
        c = torch.cat((a, b), 0)
        d = torch.cat((c, c), 1)
        return torch.cat((d, d), 2)

    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)

    jitVsGlow(test_f, x, y, expected_fused_ops={"prim::FusedConcat"})
