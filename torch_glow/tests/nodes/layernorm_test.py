from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests import utils


class SimpleLayerNormModule(torch.nn.Module):
    def __init__(self, normalized_shape):
        super(SimpleLayerNormModule, self).__init__()
        self.normalized_shape = normalized_shape

    def forward(self, input, weight=None, bias=None):
        return F.layer_norm(input, self.normalized_shape, weight, bias)


class TestLayerNorm(unittest.TestCase):
    def test_layernorm_basic(self):
        """Basic test of the PyTorch layernorm Node on Glow."""

        inputs = torch.randn(1, 4, 5, 5)
        weight = torch.randn(5)
        bias = torch.randn(5)

        utils.compare_tracing_methods(
            SimpleLayerNormModule([5]),
            inputs,
            weight,
            bias,
            fusible_ops={"aten::layer_norm"},
        )

    def test_layernorm_no_bias(self):
        """Test of the PyTorch aten::layer_norm without weights and bias."""

        inputs = torch.randn(1, 4, 5, 5)

        utils.compare_tracing_methods(
            SimpleLayerNormModule([5, 5]), inputs, fusible_ops={"aten::layer_norm"}
        )
