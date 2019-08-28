from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow

import pytest


@pytest.mark.skip(reason="incompatible with linear fusion")
def test_addmm():

    def test_f(M, aa, b, bias):
        """Basic test of the PyTorch addmm Node on Glow."""
        a = aa.t()
        M1 = M.addmm(b, a)
        return M1.add(bias)

    x = torch.randn(6, 10)
    y = torch.randn(6, 10)
    z = torch.randn(6, 6)
    a = torch.randn(6, 6)

    jitVsGlow(test_f, z, x, y, a, expected_fused_ops={"aten::linear"})
