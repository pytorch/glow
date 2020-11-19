from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleQuantizedAddModule(torch.nn.Module):
    def __init__(self, left_quantization, right_quantization, scale, zero_point):
        super(SimpleQuantizedAddModule, self).__init__()
        self.left_quantization = left_quantization
        self.right_quantization = right_quantization
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, left, right):
        return torch.nn.quantized.DeQuantize()(
            torch.ops.quantized.add(
                self.left_quantization(left),
                self.right_quantization(right),
                scale=self.scale,
                zero_point=self.zero_point,
            )
        )


class TestQuantizedAdd(unittest.TestCase):
    def test_quantized_add_zerooffset(self):
        """Basic test of the PyTorch quantized::add Node on Glow with zero offset."""

        utils.compare_tracing_methods(
            SimpleQuantizedAddModule(
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
            fusible_ops={
                "quantized::add",
                "aten::quantize_per_tensor",
                "aten::dequantize",
            },
            skip_to_glow=True,
        )

    def test_quantized_add(self):
        """Basic test of the PyTorch quantized::add Node on Glow."""

        utils.compare_tracing_methods(
            SimpleQuantizedAddModule(
                torch.nn.quantized.Quantize(
                    scale=1.0 / 128, zero_point=5, dtype=torch.quint8
                ),
                torch.nn.quantized.Quantize(
                    scale=1.0 / 128, zero_point=10, dtype=torch.quint8
                ),
                1.0 / 128,
                3,
            ),
            torch.randn([5, 5]),
            torch.randn([5, 5]),
            fusible_ops={
                "quantized::add",
                "aten::quantize_per_tensor",
                "aten::dequantize",
            },
            skip_to_glow=True,
        )

    def test_quantized_add_cut_q_dq(self):
        """Basic test of the PyTorch quantized::add Node on Glow, with quantize and dequantize excluded. """

        utils.compare_tracing_methods(
            SimpleQuantizedAddModule(
                torch.nn.quantized.Quantize(
                    scale=1.0 / 128, zero_point=5, dtype=torch.quint8
                ),
                torch.nn.quantized.Quantize(
                    scale=1.0 / 128, zero_point=10, dtype=torch.quint8
                ),
                1.0 / 128,
                3,
            ),
            torch.randn([5, 5]),
            torch.randn([5, 5]),
            fusible_ops={"quantized::add"},
            fusion_blocklist=["aten::quantize_per_tensor", "aten::dequantize"],
            skip_to_glow=True,
        )
