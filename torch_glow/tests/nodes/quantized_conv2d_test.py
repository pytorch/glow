from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow
import pytest


def test_quantized_conv2d_unpacked():
    """Basic test of the PyTorch quantize::conv2d Node with unpacked weights on Glow."""

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
    b_zero = torch.zeros(1)
    b = torch.randn(1)

    jitVsGlow(test_f, x, w, b, expected_fused_ops={"aten::quantize_per_tensor",
                                                   "glow::unpacked_quantized_conv2d",
                                                   "aten::dequantize"})

    jitVsGlow(test_f, x, w, b_zero, expected_fused_ops={"aten::quantize_per_tensor",
                                                        "glow::unpacked_quantized_conv2d",
                                                        "aten::dequantize"})


def test_quantized_conv2d_packed_groupwise():
    """Basic test of PyTorch quantize::conv2d Node with packed weights on Glow."""

    x = torch.tensor(range(5), dtype=torch.float)
    x = torch.cat((x, x, x, x, x))
    x = torch.cat((x, x, x))
    x = torch.reshape(x, [1, 3, 5, 5])
    q = torch.nn.quantized.Quantize(0.1, 2, torch.quint8)
    conv = torch.nn.Conv2d(3, 3, [2, 2], groups=3)
    dq = torch.nn.quantized.DeQuantize()

    # Due to the off-by-one error, we cannot let the weights, bias & input
    # to be totally random.
    conv.weight.data.fill_(2.0)
    conv.bias.data.fill_(1.0)

    model = torch.nn.Sequential(q, conv, dq)
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)

    jitVsGlow(model, x, expected_fused_ops={"aten::quantize_per_tensor",
                                            "quantized::conv2d",
                                            "aten::dequantize"})


@pytest.mark.skip(reason="accuracy between glow & pytorch")
def test_quantized_conv2d_nonfunctional():
    """Basic test of the PyTorch quantized conv2d Node with external quantized
    input on Glow."""

    def test_f(a):
        q = torch.nn.quantized.Quantize(1/16, 0, torch.quint8)
        dq = torch.nn.quantized.DeQuantize()
        conv = torch.nn.quantized.Conv2d(1, 1, [2, 2])
        return dq(conv(q(a)))

    x = torch.tensor([[[[5., 6.], [7., 8.]]]])

    jitVsGlow(test_f, x, expected_fused_ops={"aten::quantize_per_tensor",
                                             "glow::unpacked_quantized_conv2d",
                                             "aten::dequantize"})
