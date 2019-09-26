from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_quantized_avgpool():
    """Basic test of the PyTorch quantized::avg_pool2d Node on Glow."""

    def test_f(a):
        q = torch.nn.quantized.Quantize(
            scale=1.0/128, zero_point=3, dtype=torch.quint8)
        dq = torch.nn.quantized.DeQuantize()
        ap = torch.nn.AvgPool2d(3)
        return dq(ap(q(a)))

    inputs = torch.randn(1, 4, 5, 5)

    jitVsGlow(test_f, inputs, expected_fused_ops={"aten::avg_pool2d",
                                                  "aten::quantize_per_tensor",
                                                  "aten::dequantize"})
