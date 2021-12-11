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

import unittest

import torch
from tests import utils


class TestQuantizedConv2d(utils.TorchGlowTestCase):
    @unittest.skip(reason="Requires freezing")
    def test_quantized_conv2d_unpacked(self):
        """Basic test of the PyTorch quantize::conv2d Node with unpacked weights on Glow."""

        class SimpleQuantizedConvModel(torch.nn.Module):
            def __init__(self):
                super(SimpleQuantizedConvModel, self).__init__()

            def forward(self, a, w, b):
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

        utils.compare_tracing_methods(
            SimpleQuantizedConvModel(),
            x,
            w,
            b,
            fusible_ops={
                "aten::quantize_per_tensor",
                "glow::unpacked_quantized_conv2d",
                "aten::dequantize",
            },
            skip_to_glow=True,
        )

        utils.compare_tracing_methods(
            SimpleQuantizedConvModel(),
            x,
            w,
            b_zero,
            fusible_ops={
                "aten::quantize_per_tensor",
                "glow::unpacked_quantized_conv2d",
                "aten::dequantize",
            },
            skip_to_glow=True,
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
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

        torch.ao.quantization.prepare(model, inplace=True)
        torch.ao.quantization.convert(model, inplace=True)

        utils.compare_tracing_methods(
            model,
            x,
            fusible_ops={
                "aten::quantize_per_tensor",
                "quantized::conv2d",
                "aten::dequantize",
            },
        )

    def test_quantized_conv2d_packed_cut_q_dq(self):
        """Basic test of PyTorch quantize::conv2d Node with packed weights on Glow, with quantize and dequantize excluded."""

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
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

        torch.ao.quantization.prepare(model, inplace=True)
        torch.ao.quantization.convert(model, inplace=True)

        utils.compare_tracing_methods(
            model,
            x,
            fusible_ops={"quantized::conv2d"},
            fusion_blocklist=["aten::quantize_per_tensor", "aten::dequantize"],
            skip_to_glow=True,
        )

    def test_quantized_conv2d_packed_channelwise(self):
        """Basic test of PyTorch quantize::conv2d Node with packed channelwise weights on Glow."""

        with torch.no_grad():
            x = torch.randn([1, 4, 4, 4])

            conv = torch.nn.Conv2d(4, 2, [2, 2], groups=1)
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
                    "quantized::conv2d",
                    "aten::dequantize",
                },
            )

    def test_quantized_conv2d_packed_channelwise_serial_qconv(self):
        """Test of serial structure PyTorch quantized::conv2d on Glow."""

        class SerialQuantizedConvModel(torch.nn.Module):
            def __init__(self):
                super(SerialQuantizedConvModel, self).__init__()
                self.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

                self.quant = torch.ao.quantization.QuantStub()

                self.conv1 = torch.nn.Conv2d(4, 4, [2, 2], groups=1)
                self.conv1.weight.random_(-1, 1)
                self.conv1.bias.data.random_(-1, 1)

                self.conv2 = torch.nn.Conv2d(4, 2, [2, 2], groups=1)
                self.conv2.weight.random_(-1, 1)
                self.conv2.bias.data.random_(-1, 1)

                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.dequant(x)
                return x

        with torch.no_grad():
            x = torch.randn([1, 4, 4, 4])
            model = SerialQuantizedConvModel()

            torch.ao.quantization.prepare(model, inplace=True)
            # Calibration
            model.forward(x)
            torch.ao.quantization.convert(model, inplace=True)

            utils.compare_tracing_methods(
                model,
                x,
                fusible_ops={
                    "aten::quantize_per_tensor",
                    "quantized::conv2d",
                    "aten::dequantize",
                },
            )

    def test_quantized_conv2d_packed_channelwise_parallel_qconv(self):
        """Test of parallel structure PyTorch quantized::conv2d on Glow."""

        class ParallelQuantizedConvModel(torch.nn.Module):
            def __init__(self):
                super(ParallelQuantizedConvModel, self).__init__()
                self.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

                self.quant = torch.ao.quantization.QuantStub()

                self.conv1 = torch.nn.Conv2d(4, 2, [2, 2], groups=1)
                self.conv1.weight.random_(-1, 1)
                self.conv1.bias.data.random_(-1, 1)

                self.conv2 = torch.nn.Conv2d(4, 2, [2, 2], groups=1)
                self.conv2.weight.random_(-1, 1)
                self.conv2.bias.data.random_(-1, 1)

                self.cat = torch.ops.quantized.cat
                self.dequant = torch.ao.quantization.DeQuantStub()
                self.dequant2 = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                x1 = self.dequant(x1)
                x2 = self.dequant2(x2)
                x = torch.cat((x1, x2), dim=0)
                return x

        with torch.no_grad():
            x = torch.randn([1, 4, 4, 4])
            model = ParallelQuantizedConvModel()

            torch.ao.quantization.prepare(model, inplace=True)
            # Calibration
            model.forward(x)
            torch.ao.quantization.convert(model, inplace=True)

            utils.compare_tracing_methods(
                model,
                x,
                fusible_ops={
                    "aten::quantize_per_tensor",
                    "quantized::conv2d",
                    "aten::dequantize",
                },
                skip_to_glow=True,
            )
