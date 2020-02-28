from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow
import unittest


class TestQuantizedRelu(unittest.TestCase):
    def test_quantized_relu(self):
        """Basic test of the PyTorch quantized::relu Node on Glow."""

        def test_f(a):
            q = torch.nn.quantized.Quantize(
                scale=1.0 / 128, zero_point=3, dtype=torch.quint8
            )
            dq = torch.nn.quantized.DeQuantize()
            re = torch.nn.quantized.ReLU()
            return dq(re(q(a)))

        x = torch.randn([5, 5])

        jitVsGlow(
            test_f,
            x,
            expected_fused_ops={
                "aten::relu",
                "aten::quantize_per_tensor",
                "aten::dequantize",
            },
        )

    def test_quantized_relu_cut_dq(self):
        """Basic test of the PyTorch quantized::relu Node on Glow, with quantize and dequantize excluded. """

        def test_f(a):
            q = torch.nn.quantized.Quantize(
                scale=1.0 / 128, zero_point=3, dtype=torch.quint8
            )
            dq = torch.nn.quantized.DeQuantize()
            re = torch.nn.quantized.ReLU()
            return dq(re(q(a)))

        x = torch.randn([5, 5])

        jitVsGlow(
            test_f,
            x,
            expected_fused_ops={
                "aten::quantize_per_tensor",
                "aten::relu",
            },
            black_list=[
                "aten::dequantize",
            ]
        )
