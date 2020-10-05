from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import torch
from tests.utils import jitVsGlow


logger = logging.getLogger("quantized conv3d test")
logger.setLevel(logging.INFO)


class TestQuantizedConv3d(unittest.TestCase):
    def test_quantized_conv3d_unpacked(self):
        """Basic test of the PyTorch quantize::conv3d Node with unpacked weights on Glow."""

        def test_f(a, w, b):
            qu = torch.nn.quantized.Quantize(1 / 16, 0, torch.quint8)
            qi = torch.nn.quantized.Quantize(1 / 16, 0, torch.qint8)
            dq = torch.nn.quantized.DeQuantize()
            conv = torch.nn.quantized.functional.conv3d
            return dq(conv(qu(a), qi(w), b))

        # TODO
        # Due to the quantization error between
        # PyTorch and Glow, we would like to use some
        # determined test data instead of random ones
        # x = torch.randn([3, 3, 3, 3])
        # w = torch.randn([3, 3, 3, 3])
        # b = torch.randn([3])

        x = torch.tensor([[[[[5.0, 6.0], [7.0, 8.0]]]]])
        w = torch.tensor([[[[[2.0]]]]])
        b_zero = torch.zeros(1)
        b = torch.randn(1)

        jitVsGlow(
            test_f,
            x,
            w,
            b,
            expected_fused_ops={
                "aten::quantize_per_tensor",
                "glow::unpacked_quantized_conv3d",
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
                "glow::unpacked_quantized_conv3d",
                "aten::dequantize",
            },
        )

    def test_quantized_conv3d_packed_groupwise(self):
        """Basic test of PyTorch quantize::conv3d Node with packed weights on Glow."""

        x = torch.tensor(range(5), dtype=torch.float)
        x = torch.cat((x, x, x, x, x))
        x = torch.cat((x, x, x))
        x = torch.cat((x, x, x))
        x = torch.reshape(x, [1, 3, 3, 5, 5])
        q = torch.nn.quantized.Quantize(0.1, 2, torch.quint8)
        conv = torch.nn.Conv3d(3, 3, kernel_size=1, stride=(1, 1, 1), groups=3)
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
                "quantized::conv3d",
                "aten::dequantize",
            },
        )

    def test_quantized_conv3d_packed_cut_q_dq(self):
        """Basic test of PyTorch quantize::conv3d Node with packed weights on Glow, with quantize and dequantize excluded."""

        x = torch.tensor(range(5), dtype=torch.float)
        x = torch.cat((x, x, x, x, x))
        x = torch.cat((x, x, x))
        x = torch.cat((x, x, x))
        x = torch.reshape(x, [1, 3, 3, 5, 5])
        q = torch.nn.quantized.Quantize(0.1, 2, torch.quint8)
        conv = torch.nn.Conv3d(3, 3, kernel_size=1, stride=(1, 1, 1), groups=3)
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
            expected_fused_ops={"quantized::conv3d"},
            black_list=["aten::quantize_per_tensor", "aten::dequantize"],
        )

    def test_quantized_conv3d_packed_channelwise(self):
        """Basic test of PyTorch quantize::conv3d Node with packed channelwise weights on Glow."""

        with torch.no_grad():
            x = torch.randn([1, 4, 4, 4, 4])

            conv = torch.nn.Conv3d(4, 2, 2, (2, 2, 2), groups=1)
            conv.weight.random_(-1, 1)
            conv.bias.data.random_(-1, 1)

            model = torch.quantization.QuantWrapper(conv)
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

            torch.quantization.prepare(model, inplace=True)
            # Calibration
            model.forward(x)
            torch.quantization.convert(model, inplace=True)

            # TODO: acuracy needs to be investigated. Average acuracy is decent
            # but some elements have errors (possibly from rounding differences)
            jitVsGlow(
                model,
                x,
                expected_fused_ops={
                    "aten::quantize_per_tensor",
                    "quantized::conv3d",
                    "aten::dequantize",
                },
            )
