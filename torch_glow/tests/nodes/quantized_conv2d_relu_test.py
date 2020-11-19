from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from collections import OrderedDict

import torch
from tests import utils


class TestQuantizedConv2dRelu(unittest.TestCase):
    def _test_quantized_conv2d_relu_packed(self, groups):
        """Basic test of PyTorch quantized::conv2d_relu Node with packed weights on Glow."""
        with torch.no_grad():
            x = torch.tensor(range(5), dtype=torch.float) / 3
            x = torch.cat((x, x, x, x, x))
            x = torch.cat((x, x, x))
            x = torch.reshape(x, [1, 3, 5, 5])
            q = torch.nn.quantized.Quantize(1, 2, torch.quint8)
            conv = torch.nn.Conv2d(3, 3, [2, 2], groups=groups)
            relu = torch.nn.ReLU()
            dq = torch.nn.quantized.DeQuantize()

            # Due to the off-by-one error, we cannot let the weights, bias & input
            # to be totally random.
            conv.weight.set_(
                torch.arange(36 / groups, dtype=torch.float).reshape(
                    [3, 3 // groups, 2, 2]
                )
                / 3
            )
            conv.bias.data.fill_(2)

            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("quantize", q),
                        ("conv1", conv),
                        ("relu1", relu),
                        ("dequantize", dq),
                    ]
                )
            )
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

            # Fuse conv and relu to conv_relu
            model = torch.quantization.fuse_modules(model, [["conv1", "relu1"]])

            torch.quantization.prepare(model, inplace=True)
            torch.quantization.convert(model, inplace=True)

            utils.compare_tracing_methods(
                model,
                x,
                fusible_ops={
                    "aten::quantize_per_tensor",
                    "quantized::conv2d_relu",
                    "aten::dequantize",
                },
                skip_to_glow=True,
            )

    def test_quantized_conv2d_relu_packed_groupwise(self):
        """PyTorch groupwise quantized::conv2d_relu Node with packed weights on Glow."""
        self._test_quantized_conv2d_relu_packed(groups=3)

    def test_quantized_conv2d_relu_packed_nongroupwise(self):
        """PyTorch vanilla quantized::conv2d_relu Node with packed weights on Glow."""
        self._test_quantized_conv2d_relu_packed(groups=1)

    def test_quantized_conv2d_relu_packed_cut_q_dq(self):
        """Basic test of PyTorch quantized::conv2d_relu Node with packed weights on Glow, with quantize and dequantize excluded. """
        with torch.no_grad():
            x = torch.tensor(range(5), dtype=torch.float) / 3
            x = torch.cat((x, x, x, x, x))
            x = torch.cat((x, x, x))
            x = torch.reshape(x, [1, 3, 5, 5])
            q = torch.nn.quantized.Quantize(1, 2, torch.quint8)
            conv = torch.nn.Conv2d(3, 3, [2, 2], groups=1)
            relu = torch.nn.ReLU()
            dq = torch.nn.quantized.DeQuantize()

            # Due to the off-by-one error, we cannot let the weights, bias & input
            # to be totally random.
            conv.weight.set_(
                torch.arange(36, dtype=torch.float).reshape([3, 3, 2, 2]) / 3
            )
            conv.bias.data.fill_(2)

            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("quantize", q),
                        ("conv1", conv),
                        ("relu1", relu),
                        ("dequantize", dq),
                    ]
                )
            )
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

            # Fuse conv and relu to conv_relu
            model = torch.quantization.fuse_modules(model, [["conv1", "relu1"]])

            torch.quantization.prepare(model, inplace=True)
            torch.quantization.convert(model, inplace=True)

            utils.compare_tracing_methods(
                model,
                x,
                fusible_ops={"quantized::conv2d_relu"},
                fusion_blocklist=["aten::quantize_per_tensor", "aten::dequantize"],
                skip_to_glow=True,
            )
