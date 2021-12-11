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

import torch
from tests import utils


class SimpleQuantizedLinearModel(torch.nn.Sequential):
    def __init__(
        self,
        in_features,
        out_features,
        quantization,
        per_tensor,
        weight=None,
        bias=None,
    ):
        linear = torch.nn.Linear(in_features, out_features, bias=(bias is not None))
        if weight:
            linear.weight.data.fill_(weight)
        else:
            linear.weight.data.random_(0, 100)
        if bias:
            linear.bias.data.fill_(bias)

        super(SimpleQuantizedLinearModel, self).__init__(
            quantization, linear, torch.nn.quantized.DeQuantize()
        )

        weight_observer = (
            torch.ao.quantization.default_weight_observer
            if per_tensor
            else torch.ao.quantization.default_per_channel_weight_observer
        )
        self.qconfig = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.default_observer,
            weight=weight_observer,
        )

        torch.ao.quantization.prepare(self, inplace=True)
        torch.ao.quantization.convert(self, inplace=True)


def _make_input(size, duplications, shape, dtype=torch.float):
    tensor = torch.tensor(range(size), dtype=dtype)
    tensor = torch.cat(tuple(tensor for _ in range(duplications)))
    tensor = torch.reshape(tensor, shape)
    return tensor


class TestQuantizedLinear(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "basic",
                SimpleQuantizedLinearModel(
                    5,
                    5,
                    torch.nn.quantized.Quantize(
                        scale=1 / 25, zero_point=17, dtype=torch.quint8
                    ),
                    False,  # per_tensor
                    1.2,
                    3.0,
                ),
                _make_input(5, 6, [3, 2, 5]),
            ),
            lambda: (
                "no_bias",
                SimpleQuantizedLinearModel(
                    5,
                    3,
                    torch.nn.quantized.Quantize(
                        scale=1 / 15, zero_point=17, dtype=torch.quint8
                    ),
                    False,  # per_tensor
                    1.2,
                ),
                _make_input(5, 6, [3, 2, 5]),
            ),
            lambda: (
                "exclude_dq",
                SimpleQuantizedLinearModel(
                    5,
                    5,
                    torch.nn.quantized.Quantize(
                        scale=1 / 25, zero_point=17, dtype=torch.quint8
                    ),
                    False,  # per_tensor
                    1.2,
                    3.0,
                ),
                _make_input(5, 6, [3, 2, 5]),
                {"aten::dequantize"},
            ),
            lambda: (
                "rowwise",
                SimpleQuantizedLinearModel(
                    6,
                    5,
                    torch.nn.quantized.Quantize(
                        scale=1 / 25, zero_point=17, dtype=torch.quint8
                    ),
                    False,  # per_tensor
                ),
                _make_input(36, 1, [3, 2, 6]),
            ),
            lambda: (
                "tensorwise",
                SimpleQuantizedLinearModel(
                    6,
                    5,
                    torch.nn.quantized.Quantize(
                        scale=1 / 25, zero_point=17, dtype=torch.quint8
                    ),
                    True,  # per_tensor
                ),
                _make_input(36, 1, [3, 2, 6]),
            ),
        ]
    )
    def test_quantized_linear(self, _, model, tensor, fusion_blocklist=None):
        fusible_ops = {
            "aten::quantize_per_tensor",
            "quantized::linear",
            "aten::dequantize",
        }
        fusible_ops -= fusion_blocklist or set()
        utils.compare_tracing_methods(
            model, tensor, fusible_ops=fusible_ops, fusion_blocklist=fusion_blocklist
        )
