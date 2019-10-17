from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_quantized_linear():
    """Basic test of the PyTorch quantized::linear Node on Glow."""

    def test_f(inputs, weights, bias=None):
        q_int = torch.nn.quantized.Quantize(
            scale=2.0, zero_point=0, dtype=torch.qint8)
        q_uint = torch.nn.quantized.Quantize(
            scale=1.5, zero_point=120, dtype=torch.quint8)

        dq = torch.nn.quantized.DeQuantize()

        q_inputs = q_uint(inputs)
        q_weights = q_int(weights)

        return dq(torch.nn.quantized.functional.linear(q_inputs, q_weights, bias))

    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = torch.tensor([[2.0, 4.0], [3.0, 1.0], [6.0, 1.0]])
    bias = torch.tensor([5.0, 3.0, 2.0])

    jitVsGlow(test_f, inputs, weights, bias, expected_fused_ops={
              "glow::unpacked_quantized_linear", "aten::quantize_per_tensor",
              "aten::dequantize"})
