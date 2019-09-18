from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow

def test_quantized_relu():
    """Basic test of the PyTorch quantized::relu Node on Glow."""

    def test_f(a):
        q = torch.nn.quantized.Quantize(0.015, 5, torch.quint8)
        dq = torch.nn.quantized.DeQuantize()
        return dq(torch.nn.quantized.ReLU(q(a)))

    x = torch.randn([5, 5])

    jitVsGlow(test_f, x, expected_fused_ops={"quantized::add",
                                                "aten::quantize_linear",
                                                "aten::dequantize"})
