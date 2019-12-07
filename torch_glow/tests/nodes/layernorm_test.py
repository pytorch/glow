from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F

from tests.utils import jitVsGlow
import unittest


class TestLayerNorm(unittest.TestCase):
    def test_layernorm_basic(self):
        """Basic test of the PyTorch layernorm Node on Glow."""

        def test_f(inputs, weight, bias):
            return F.layer_norm(inputs, [5], weight, bias)

        inputs = torch.randn(1, 4, 5, 5)
        weight = torch.randn(5)
        bias = torch.randn(5)

        jitVsGlow(test_f, inputs, weight, bias,
                  expected_fused_ops={"aten::layer_norm"})

    def test_layernorm_no_bias(self):
        """Test of the PyTorch aten::layer_norm without weights and bias."""

        def test_f(inputs):
            return F.layer_norm(inputs, [5, 5])

        inputs = torch.randn(1, 4, 5, 5)

        jitVsGlow(test_f, inputs, expected_fused_ops={"aten::layer_norm"})
