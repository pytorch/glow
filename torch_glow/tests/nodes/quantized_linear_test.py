import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleQuantizedLinearModel(torch.nn.Sequential):
    def __init__(self, in_features, out_features, quantization, weight=None, bias=None):
        linear = torch.nn.Linear(in_features, out_features)
        if weight:
            linear.weight.data.fill_(weight)
        else:
            linear.weight.data.random_(0, 100)
        if bias:
            linear.bias.data.fill_(bias)
        else:
            linear.bias.data.random_(0, 10)
        super(SimpleQuantizedLinearModel, self).__init__(
            quantization, linear, torch.nn.quantized.DeQuantize()
        )
        self.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(self, inplace=True)
        torch.quantization.convert(self, inplace=True)


def _make_input(size, duplications, shape, dtype=torch.float):
    tensor = torch.tensor(range(size), dtype=dtype)
    tensor = torch.cat(tuple(tensor for _ in range(duplications)))
    tensor = torch.reshape(tensor, shape)
    return tensor


class TestQuantizedLinear(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic",
                SimpleQuantizedLinearModel(
                    5,
                    5,
                    torch.nn.quantized.Quantize(
                        scale=1 / 25, zero_point=17, dtype=torch.quint8
                    ),
                    1.2,
                    3.0,
                ),
                _make_input(5, 6, [3, 2, 5]),
            ),
            (
                "exclude_dq",
                SimpleQuantizedLinearModel(
                    5,
                    5,
                    torch.nn.quantized.Quantize(
                        scale=1 / 25, zero_point=17, dtype=torch.quint8
                    ),
                    1.2,
                    3.0,
                ),
                _make_input(5, 6, [3, 2, 5]),
                {"aten::dequantize"},
            ),
            (
                "rowwise",
                SimpleQuantizedLinearModel(
                    6,
                    5,
                    torch.nn.quantized.Quantize(
                        scale=1 / 25, zero_point=17, dtype=torch.quint8
                    ),
                ),
                _make_input(36, 1, [3, 2, 6]),
                {"aten::dequantize"},
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
