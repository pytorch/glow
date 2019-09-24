from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_quantized_add_zerooffset():
    """Basic test of the PyTorch quantized::add Node on Glow with zero offset."""

    def test_f(a, b):
        q = torch.nn.quantized.Quantize(0.3, 0, torch.quint8)
        dq = torch.nn.quantized.DeQuantize()
        return dq(torch.ops.quantized.add(q(a), q(b), scale=0.05, zero_point=0))

    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    y = torch.tensor([5, 6, 7, 8], dtype=torch.float32)

    jitVsGlow(test_f, x, y, expected_fused_ops={"quantized::add",
                                                "aten::quantize_per_tensor",
                                                "aten::dequantize"})


def test_quantized_add():
    """Basic test of the PyTorch quantized::add Node on Glow."""

    def test_f(a, b):
        q1 = torch.nn.quantized.Quantize(1/128, 5, torch.quint8)
        q2 = torch.nn.quantized.Quantize(1/128, 10, torch.quint8)
        dq = torch.nn.quantized.DeQuantize()
        return dq(torch.ops.quantized.add(q1(a), q2(b), scale=1/128, zero_point=3))

    x = torch.randn([5, 5])
    y = torch.randn([5, 5])

    jitVsGlow(test_f, x, y, expected_fused_ops={"quantized::add",
                                                "aten::quantize_per_tensor",
                                                "aten::dequantize"})
