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

from collections import OrderedDict

import torch
from tests import utils


class TestQuantizedConv3dRelu(utils.TorchGlowTestCase):
    def _test_quantized_conv3d_relu_packed(self, groups):
        """Basic test of PyTorch quantized::conv3d_relu Node with packed weights on Glow."""
        with torch.no_grad():
            x = torch.tensor(range(5), dtype=torch.float)
            x = torch.cat((x, x, x, x, x))
            x = torch.cat((x, x, x))
            x = torch.cat((x, x, x))
            x = torch.reshape(x, [1, 3, 3, 5, 5])
            q = torch.nn.quantized.Quantize(1, 2, torch.quint8)
            conv = torch.nn.Conv3d(3, 3, kernel_size=3, stride=(2, 2, 2), groups=groups)
            relu = torch.nn.ReLU()
            dq = torch.nn.quantized.DeQuantize()

            # Due to the off-by-one error, we cannot let the weights, bias & input
            # to be totally random.
            conv.weight.set_(
                torch.arange(72 / groups, dtype=torch.float).reshape(
                    [3, 3 // groups, 2, 2, 2]
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
            model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

            # Fuse conv and relu to conv_relu
            model = torch.ao.quantization.fuse_modules(model, [["conv1", "relu1"]])

            torch.ao.quantization.prepare(model, inplace=True)
            torch.ao.quantization.convert(model, inplace=True)

            utils.compare_tracing_methods(
                model,
                x,
                fusible_ops={
                    "aten::quantize_per_tensor",
                    "quantized::conv3d_relu",
                    "aten::dequantize",
                },
                skip_to_glow=True,
            )

    def test_quantized_conv3d_relu_packed_groupwise(self):
        """PyTorch groupwise quantized::conv3d_relu Node with packed weights on Glow."""
        self._test_quantized_conv3d_relu_packed(groups=3)

    def test_quantized_conv3d_relu_packed_nongroupwise(self):
        """PyTorch vanilla quantized::conv3d_relu Node with packed weights on Glow."""
        self._test_quantized_conv3d_relu_packed(groups=1)

    def test_quantized_conv3d_relu_packed_cut_q_dq(self):
        """Basic test of PyTorch quantized::conv3d_relu Node with packed weights on Glow, with quantize and dequantize excluded."""
        with torch.no_grad():
            x = torch.tensor(range(5), dtype=torch.float)
            x = torch.cat((x, x, x, x, x))
            x = torch.cat((x, x, x))
            x = torch.cat((x, x, x))
            x = torch.reshape(x, [1, 3, 3, 5, 5])
            q = torch.nn.quantized.Quantize(1, 2, torch.quint8)
            conv = torch.nn.Conv3d(3, 3, kernel_size=3, stride=(2, 2, 2), groups=1)
            relu = torch.nn.ReLU()
            dq = torch.nn.quantized.DeQuantize()

            # Due to the off-by-one error, we cannot let the weights, bias & input
            # to be totally random.
            conv.weight.set_(
                torch.arange(72, dtype=torch.float).reshape([3, 3, 2, 2, 2]) / 3
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
            model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

            # Fuse conv and relu to conv_relu
            model = torch.ao.quantization.fuse_modules(model, [["conv1", "relu1"]])

            torch.ao.quantization.prepare(model, inplace=True)
            torch.ao.quantization.convert(model, inplace=True)

            utils.compare_tracing_methods(
                model,
                x,
                fusible_ops={"quantized::conv3d_relu"},
                fusion_blocklist=["aten::quantize_per_tensor", "aten::dequantize"],
                skip_to_glow=True,
            )
