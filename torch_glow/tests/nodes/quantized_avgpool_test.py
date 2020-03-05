from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow
import unittest


class TestQuantizedAvgPool(unittest.TestCase):
    def test_quantized_avgpool(self):
        """Basic test of the PyTorch quantized::avg_pool2d Node on Glow."""

        def test_f(a):
            q = torch.nn.quantized.Quantize(
                scale=1.0 / 128, zero_point=3, dtype=torch.quint8
            )
            dq = torch.nn.quantized.DeQuantize()
            ap = torch.nn.AvgPool2d(3)
            return dq(ap(q(a)))

        inputs = torch.randn(1, 4, 5, 5)

        jitVsGlow(
            test_f,
            inputs,
            expected_fused_ops={
                "aten::avg_pool2d",
                "aten::quantize_per_tensor",
                "aten::dequantize",
            },
        )

    def test_quantized_avgpool_cut_q_dq(self):
        """Basic test of the PyTorch quantized::avg_pool2d Node on Glow, with quantize and dequantize excluded. """

        def test_f(a):
            q = torch.nn.quantized.Quantize(
                scale=1.0 / 128, zero_point=3, dtype=torch.quint8
            )
            dq = torch.nn.quantized.DeQuantize()
            ap = torch.nn.AvgPool2d(3)
            return dq(ap(q(a)))

        inputs = torch.randn(1, 4, 5, 5)

        jitVsGlow(
            test_f,
            inputs,
            expected_fused_ops={
                "aten::avg_pool2d",
            },
            black_list=[
                "aten::quantize_per_tensor",
                "aten::dequantize",
            ]
        )
