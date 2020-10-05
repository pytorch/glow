from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests.utils import jitVsGlow


class TestLinear(unittest.TestCase):
    def test_linear_basic(self):
        """Basic test of the PyTorch aten::linear op on Glow."""

        def test_f(input, weight, bias=None):
            return F.linear((input + input), weight, bias)

        n = 5
        in_features = 4
        out_features = 3

        input = torch.randn(n, in_features)
        weight = torch.randn(out_features, in_features)

        # expected_fused_ops has is empty because linear gets lowered to other ops
        jitVsGlow(test_f, input, weight, expected_fused_ops={})

    def test_linear_bias(self):
        """Test of the PyTorch aten::linear op on Glow."""

        def test_f(input, weight, bias=None):
            return F.linear((input + input), weight, bias)

        n = 5
        in_features = 4
        out_features = 3

        input = torch.randn(n, in_features)
        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)

        # expected_fused_ops has is empty because linear gets lowered to other ops
        jitVsGlow(test_f, input, weight, bias, expected_fused_ops={})

    def test_linear_broadcast(self):
        """Test of the PyTorch aten::linear op with broadcasting on Glow."""

        def test_f(input, weight, bias=None):
            return F.linear((input + input), weight, bias)

        n = 5
        in_features = 4
        out_features = 3

        input = torch.randn(n, 9, 7, in_features)
        weight = torch.randn(out_features, in_features)

        # expected_fused_ops has is empty because linear gets lowered to other ops
        jitVsGlow(test_f, input, weight, expected_fused_ops={})
