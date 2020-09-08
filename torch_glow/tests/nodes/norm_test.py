from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestNorm(unittest.TestCase):
    def test_norm_basic(self):
        """Basic test of the PyTorch norm Node on Glow."""

        def test_f(a):
            return torch.norm(a, dim=0, p=2)

        a = torch.arange(8, dtype=torch.float).reshape(2, 4)

        jitVsGlow(test_f, a, expected_fused_ops={"aten::norm"})

    def test_norm_3d_inner_axis(self):
        """Basic test of the PyTorch norm Node on Glow."""

        def test_f(a):
            return torch.norm(a, dim=1)

        a = torch.arange(8, dtype=torch.float).reshape(2, 2, 2)

        jitVsGlow(test_f, a, expected_fused_ops={"aten::frobenius_norm"})

    def test_norm_4d_outer_axis(self):
        """Basic test of the PyTorch norm Node on Glow."""

        def test_f(a):
            return torch.norm(a, dim=[3])

        a = torch.arange(16, dtype=torch.float).reshape(2, 2, 2, 2)

        jitVsGlow(test_f, a, expected_fused_ops={"aten::frobenius_norm"})
