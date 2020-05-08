from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
import torch_glow
from tests.utils import jitVsGlow


class TestJITVsGlowPath(unittest.TestCase):
    def test_jit_vs_glow_path(self):
        """Basic test of the JIT vs. Glow logging feature."""

        torch_glow.enable_jit_vs_glow_compare()

        def test_f(input, weight, bias=None):
            return F.linear((input + input), weight, bias)

        n = 5
        in_features = 4
        out_features = 3

        input = torch.randn(n, in_features)
        weight = torch.randn(out_features, in_features)

        jitVsGlow(
            test_f,
            input,
            weight,
            expected_fused_ops={"aten::add", "aten::t", "aten::matmul"},
        )

    def test_jit_vs_glow_int_path(self):
        """Test JIT vs. Glow logging with int type """

        torch_glow.enable_jit_vs_glow_compare()

        def test_f(a, b):
            c = a + b
            return c

        a = torch.randn(5, 6).to(dtype=torch.int32)
        b = torch.randn(5, 6).to(dtype=torch.int32)

        jitVsGlow(test_f, a, b, expected_fused_ops={"aten::add"})

    def test_jit_vs_glow_inplace(self):
        """Test JIT vs. Glow logging with in-place op"""

        torch_glow.enable_jit_vs_glow_compare()

        def test_f(a, b):
            a += b
            return a

        a = torch.randn(5, 6)
        b = torch.randn(5, 6)

        jitVsGlow(test_f, a, b, expected_fused_ops={"aten::add_"})
