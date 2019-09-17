from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_quantization_basic():
    """Basic test of the PyTorch quantize and dequantize Node on Glow."""

    # This only test if operator has been successfully projected
    # Calculating process in glow will be optimized
    # TODO:zrphercule add aten::add once add is supported
    def test_simple(a, b):
        q = torch.nn.quantized.Quantize(1, 0, torch.quint8)
        dq = torch.nn.quantized.DeQuantize()
        return dq(q(a.add(b)))

    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    y = torch.tensor([5, 6, 7, 8], dtype=torch.float32)

    jitVsGlow(test_simple, x, y, expected_fused_ops={"aten::add",
                                                     "aten::quantize_linear",
                                                     "aten::dequantize"})
