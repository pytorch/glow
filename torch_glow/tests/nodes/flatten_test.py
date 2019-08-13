from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests.utils import jitVsGlow


def test_flatten_basic():
    """Test of the PyTorch flatten Node on Glow."""

    def flatten_basic(a):
        return torch.flatten(a, 1)

    x = torch.randn(2, 2, 2)

    jitVsGlow(flatten_basic, x, expected_fused_ops={"aten::flatten"})
