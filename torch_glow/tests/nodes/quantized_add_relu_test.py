from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleQuantizedAddReluModule(torch.nn.Module):
    def __init__(self, left_quantization, right_quantization, scale, zero_point):
        super(SimpleQuantizedAddReluModule, self).__init__()
        self.left_quantization = left_quantization
        self.right_quantization = right_quantization
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, left, right):
        return torch.nn.quantized.DeQuantize()(
            torch.ops.quantized.add_relu(
                self.left_quantization(left),
                self.right_quantization(right),
                scale=self.scale,
                zero_point=self.zero_point,
            )
        )


class TestQuantizedAddRelu(unittest.TestCase):
    def test_quantized_add_relu_zerooffset(self):
        """Basic test of the PyTorch quantized::add Node_relu on Glow with zero offset."""

        utils.compare_tracing_methods(
            SimpleQuantizedAddReluModule(
                torch.nn.quantized.Quantize(
                    scale=0.3, zero_point=0, dtype=torch.quint8
                ),
                torch.nn.quantized.Quantize(
                    scale=0.3, zero_point=0, dtype=torch.quint8
                ),
                0.05,
                0,
            ),
            torch.tensor([1, 2, 3, 4], dtype=torch.float32),
            torch.tensor([5, 6, 7, 8], dtype=torch.float32),
            skip_to_glow=True,
        )

    def test_quantized_add_relu(self):
        """Basic test of the PyTorch quantized::add_relu Node on Glow."""

        utils.compare_tracing_methods(
            SimpleQuantizedAddReluModule(
                torch.nn.quantized.Quantize(
                    scale=1.0 / 128, zero_point=5, dtype=torch.quint8
                ),
                torch.nn.quantized.Quantize(
                    scale=1.0 / 128, zero_point=10, dtype=torch.quint8
                ),
                1.0 / 128,
                3,
            ),
            torch.rand([5, 5]),
            torch.rand([5, 5]),
            skip_to_glow=True,
        )

    def test_quantized_add_relu_cut_q_dq(self):
        """Basic test of the PyTorch quantized::add_relu Node on Glow, with quantize and dequantize excluded. """

        utils.compare_tracing_methods(
            SimpleQuantizedAddReluModule(
                torch.nn.quantized.Quantize(
                    scale=1.0 / 128, zero_point=5, dtype=torch.quint8
                ),
                torch.nn.quantized.Quantize(
                    scale=1.0 / 128, zero_point=10, dtype=torch.quint8
                ),
                1.0 / 128,
                3,
            ),
            torch.rand([5, 5]),
            torch.rand([5, 5]),
            fusible_ops={"quantized::add_relu"},
            fusion_blocklist=["aten::quantize_per_tensor", "aten::dequantize"],
            skip_to_glow=True,
        )
