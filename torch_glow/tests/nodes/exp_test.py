from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_exp_basic():
    """Test of the PyTorch exp Node on Glow."""

    def test_f(a):
        b = torch.exp(a)
        return torch.exp(b)

    x = torch.randn(4)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::exp"})
