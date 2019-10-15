from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_quantized_conv2d():
    """Basic test of the PyTorch quantized::relu Node on Glow."""

    def test_f(a, w, b):
        qu = torch.nn.quantized.Quantize(1/16, 0, torch.quint8)
        qi = torch.nn.quantized.Quantize(1/16, 0, torch.qint8)
        dq = torch.nn.quantized.DeQuantize()
        conv = torch.nn.quantized.functional.conv2d
        return dq(conv(qu(a), qi(w), b))

    # TODO
    # Due to the quantization error between
    # PyTorch and Glow, we would like to use some
    # determined test data instead of random ones
    # x = torch.randn([3, 3, 3, 3])
    # w = torch.randn([3, 3, 3, 3])
    # b = torch.randn([3])

    x = torch.tensor([[[[5., 6.], [7., 8.]]]])
    w = torch.tensor([[[[1., 2.], [3., 4.]]]])
    b = torch.zeros(1)

    jitVsGlow(test_f, x, w, b, expected_fused_ops={"aten::quantize_per_tensor",
                                                   "glow::unpacked_quantized_conv2d",
                                                   "aten::dequantize"})
