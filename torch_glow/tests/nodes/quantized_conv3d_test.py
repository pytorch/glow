# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import torch
from tests import utils


logger = logging.getLogger("quantized conv3d test")
logger.setLevel(logging.INFO)


class UnpackedConv3dModel(torch.nn.Module):
    def __init__(self, input_quantization, weight_quantization):
        super(UnpackedConv3dModel, self).__init__()
        self.input_quantization = input_quantization
        self.weight_quantization = weight_quantization

    def forward(self, tensor, weight, bias):
        return torch.nn.quantized.DeQuantize()(
            torch.nn.quantized.functional.conv3d(
                self.input_quantization(tensor), self.weight_quantization(weight), bias
            )
        )


class PackedConv3dModel(torch.nn.Sequential):
    def __init__(self, quantization, convolution, dequantization, weight, bias):
        # Due to the off-by-one error, we cannot let the weights, bias & input
        # to be totally random.
        convolution.weight.data.fill_(weight)
        convolution.bias.data.fill_(bias)
        super(PackedConv3dModel, self).__init__(
            quantization, convolution, dequantization
        )
        self.eval()
        self.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        torch.ao.quantization.prepare(self, inplace=True)
        torch.ao.quantization.convert(self, inplace=True)


class TestQuantizedConv3d(utils.TorchGlowTestCase):
    @unittest.skip(reason="Requires freezing")
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

        utils.compare_tracing_methods(
            UnpackedConv3dModel(
                torch.nn.quantized.Quantize(1 / 16, 0, torch.quint8),
                torch.nn.quantized.Quantize(1 / 16, 0, torch.qint8),
            ),
            x,
            w,
            b,
            fusible_ops={
                "aten::quantize_per_tensor",
                "glow::unpacked_quantized_conv3d",
                "aten::dequantize",
            },
            skip_to_glow=True,
        )

        utils.compare_tracing_methods(
            UnpackedConv3dModel(
                torch.nn.quantized.Quantize(1 / 16, 0, torch.quint8),
                torch.nn.quantized.Quantize(1 / 16, 0, torch.qint8),
            ),
            x,
            w,
            b_zero,
            fusible_ops={
                "aten::quantize_per_tensor",
                "glow::unpacked_quantized_conv3d",
                "aten::dequantize",
            },
            skip_to_glow=True,
        )

    def test_quantized_conv3d_packed_groupwise(self):
        """Basic test of PyTorch quantize::conv3d Node with packed weights on Glow."""

        x = torch.tensor(range(5), dtype=torch.float)
        x = torch.cat((x, x, x, x, x))
        x = torch.cat((x, x, x))
        x = torch.cat((x, x, x))
        x = torch.reshape(x, [1, 3, 3, 5, 5])

        utils.compare_tracing_methods(
            PackedConv3dModel(
                torch.nn.quantized.Quantize(0.1, 2, torch.quint8),
                torch.nn.Conv3d(3, 3, kernel_size=1, stride=(1, 1, 1), groups=3),
                torch.nn.quantized.DeQuantize(),
                2.0,
                1.0,
            ),
            x,
            fusible_ops={
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

        utils.compare_tracing_methods(
            PackedConv3dModel(
                torch.nn.quantized.Quantize(0.1, 2, torch.quint8),
                torch.nn.Conv3d(3, 3, kernel_size=1, stride=(1, 1, 1), groups=3),
                torch.nn.quantized.DeQuantize(),
                2.0,
                1.0,
            ),
            x,
            fusible_ops={"quantized::conv3d"},
            fusion_blocklist=["aten::quantize_per_tensor", "aten::dequantize"],
        )

    def test_quantized_conv3d_packed_channelwise(self):
        """Basic test of PyTorch quantize::conv3d Node with packed channelwise weights on Glow."""

        with torch.no_grad():
            x = torch.randn([1, 4, 4, 4, 4])

            conv = torch.nn.Conv3d(4, 2, 2, (2, 2, 2), groups=1)
            conv.weight.random_(-1, 1)
            conv.bias.data.random_(-1, 1)

            model = torch.ao.quantization.QuantWrapper(conv)
            model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

            torch.ao.quantization.prepare(model, inplace=True)
            # Calibration
            model.forward(x)
            torch.ao.quantization.convert(model, inplace=True)

            # TODO: acuracy needs to be investigated. Average acuracy is decent
            # but some elements have errors (possibly from rounding differences)
            utils.compare_tracing_methods(
                model,
                x,
                fusible_ops={
                    "aten::quantize_per_tensor",
                    "quantized::conv3d",
                    "aten::dequantize",
                },
            )
