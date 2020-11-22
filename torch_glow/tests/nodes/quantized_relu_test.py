from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleQuantizedReluModel(torch.nn.Module):
    def __init__(self, scale, zero_point, dtype):
        super(SimpleQuantizedReluModel, self).__init__()
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    def forward(self, tensor):
        quantize = torch.nn.quantized.Quantize(
            scale=self.scale, zero_point=self.zero_point, dtype=self.dtype
        )
        dequantize = torch.nn.quantized.DeQuantize()
        relu = torch.nn.ReLU()
        return dequantize(relu(quantize(tensor)))


class TestQuantizedRelu(unittest.TestCase):
    def test_quantized_relu(self):
        """Basic test of the PyTorch quantized::relu Node on Glow."""

        utils.compare_tracing_methods(
            SimpleQuantizedReluModel(1.0 / 128, 3, torch.quint8),
            torch.randn([5, 5]),
            fusible_ops={"aten::relu", "aten::quantize_per_tensor", "aten::dequantize"},
        )

    def test_quantized_relu_cut_dq(self):
        """Basic test of the PyTorch quantized::relu Node on Glow, with quantize and dequantize excluded. """

        utils.compare_tracing_methods(
            SimpleQuantizedReluModel(1.0 / 128, 3, torch.quint8),
            torch.randn([5, 5]),
            fusible_ops={"aten::relu", "aten::quantize_per_tensor"},
            fusion_blocklist=["aten::dequantize"],
        )
