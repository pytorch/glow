from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow

import pytest


@pytest.mark.skip(reason="not ready")
def test_addmm():

    def test_f(M, aa, b, bias):
        """Basic test of the PyTorch addmm Node on Glow."""
        a = aa.t()
        M1 = M.addmm(a, b)
        M2 = M1.add(bias)
        return M2

    x = torch.randn(10, 6)
    y = torch.randn(10, 6)
    z = torch.randn(6, 6)
    a = torch.randn(6, 6)

    jitVsGlow(test_f, z, x, y, a, expected_fused_ops={"aten::linear"})
