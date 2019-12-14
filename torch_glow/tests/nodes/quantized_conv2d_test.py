from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow
import unittest


class TestQuantizedConv2d(unittest.TestCase):
    def test_quantized_conv2d_unpacked(self):
        """Basic test of the PyTorch quantize::conv2d Node with unpacked weights on Glow."""

        def test_f(a, w, b):
            qu = torch.nn.quantized.Quantize(1 / 16, 0, torch.quint8)
            qi = torch.nn.quantized.Quantize(1 / 16, 0, torch.qint8)
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

        x = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]])
        w = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        b_zero = torch.zeros(1)
        b = torch.randn(1)

        jitVsGlow(
            test_f,
            x,
            w,
            b,
            expected_fused_ops={
                "aten::quantize_per_tensor",
                "glow::unpacked_quantized_conv2d",
                "aten::dequantize",
            },
        )

        jitVsGlow(
            test_f,
            x,
            w,
            b_zero,
            expected_fused_ops={
                "aten::quantize_per_tensor",
                "glow::unpacked_quantized_conv2d",
                "aten::dequantize",
            },
        )

    def test_quantized_conv2d_packed_groupwise(self):
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
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)

        jitVsGlow(
            model,
            x,
            expected_fused_ops={
                "aten::quantize_per_tensor",
                "quantized::conv2d",
                "aten::dequantize",
            },
        )

    def test_quantized_conv2d_packed_channelwise(self):
        """Basic test of PyTorch quantize::conv2d Node with packed channelwise weights on Glow."""

        with torch.no_grad():
            x = torch.tensor(range(64), dtype=torch.float) - 32
            x = x.reshape([1, 4, 4, 4])

            conv = torch.nn.Conv2d(4, 2, [2, 2], groups=1)
            conv.weight.set_(torch.arange(32,
                                          dtype=torch.float).reshape([2, 4, 2, 2])-10)
            conv.bias.data.random_(-5, 5)

            model = torch.quantization.QuantWrapper(conv)
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

            torch.quantization.prepare(model, inplace=True)
            # Calibration
            model.forward(x)
            torch.quantization.convert(model, inplace=True)

            jitVsGlow(
                model,
                x,
                expected_fused_ops={
                    "aten::quantize_per_tensor",
                    "quantized::conv2d",
                    "aten::dequantize",
                },
            )
