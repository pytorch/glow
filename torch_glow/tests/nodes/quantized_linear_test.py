from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_quantized_linear_random_input():
    """Basic test of the PyTorch quantized::linear Node on Glow."""

    def test_f(inputs, weights, bias=None):
        q_int = torch.nn.quantized.Quantize(
            scale=1/13, zero_point=0, dtype=torch.qint8)
        q_uint = torch.nn.quantized.Quantize(
            scale=1/13, zero_point=10, dtype=torch.quint8)

        dq = torch.nn.quantized.DeQuantize()

        q_inputs = q_uint(inputs)
        q_weights = q_int(weights)

        return dq(torch.nn.quantized.functional.linear(q_inputs, q_weights, bias))

    for _ in range(100):
        inputs = torch.randn(7, 7)
        weights = torch.randn(7, 7)

        bias = torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.float) * 0.1

        jitVsGlow(test_f, inputs, weights, bias, expected_fused_ops={
            "glow::unpacked_quantized_linear", "aten::quantize_per_tensor",
            "aten::dequantize"})
