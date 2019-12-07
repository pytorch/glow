from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow
import pytest

from collections import OrderedDict


def test_quantized_conv2d_relu_packed_groupwise():
    """Basic test of PyTorch quantized::conv2d_relu Node with packed weights on Glow."""

    x = torch.tensor(range(5), dtype=torch.float) * 1.5
    x = torch.cat((x, x, x, x, x))
    x = torch.cat((x, x, x))
    x = torch.reshape(x, [1, 3, 5, 5])
    q = torch.nn.quantized.Quantize(0.2, 2, torch.quint8)
    conv = torch.nn.Conv2d(3, 3, [2, 2], groups=3)
    relu = torch.nn.ReLU()
    dq = torch.nn.quantized.DeQuantize()

    # Due to the off-by-one error, we cannot let the weights, bias & input
    # to be totally random.
    conv.weight.data.fill_(1.5)
    conv.bias.data.fill_(2.5)

    model = torch.nn.Sequential(OrderedDict([
        ('quantize', q),
        ('conv1', conv),
        ('relu1', relu),
        ('deuantize', dq)]))
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Fuse conv and relu to conv_relu
    model = torch.quantization.fuse_modules(model, [['conv1', 'relu1']])

    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)

    jitVsGlow(model, x, expected_fused_ops={"aten::quantize_per_tensor",
                                            "quantized::conv2d_relu",
                                            "aten::dequantize"})
