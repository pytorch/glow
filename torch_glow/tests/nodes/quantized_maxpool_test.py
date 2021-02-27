from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleQuantizedMaxPoolModel(torch.nn.Module):
    def __init__(self, scale, zero_point, dtype, kernel_size):
        super(SimpleQuantizedMaxPoolModel, self).__init__()
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    def forward(self, tensor):
        quantize = torch.nn.quantized.Quantize(
            scale=self.scale, zero_point=self.zero_point, dtype=self.dtype
        )
        dequantize = torch.nn.quantized.DeQuantize()
        maxpool = torch.nn.MaxPool2d(3)
        dequantize = torch.nn.quantized.DeQuantize()
        return dequantize(maxpool(quantize(tensor)))


class TestQuantizedMaxPool(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "basic",
                SimpleQuantizedMaxPoolModel(1.0 / 128, 3, torch.quint8, 3),
                torch.randn(1, 4, 5, 5),
            ),
            lambda: (
                "cut_q",
                SimpleQuantizedMaxPoolModel(1.0 / 128, 3, torch.quint8, 3),
                torch.randn(1, 4, 5, 5),
                {"aten::quantize_per_tensor"},
            ),
        ]
    )
    def test_quantized_maxpool(self, _, module, tensor, fusion_blocklist=None):
        fusible_ops = {
            "aten::max_pool2d",
            "aten::quantize_per_tensor",
            "aten::dequantize",
        }
        fusible_ops -= fusion_blocklist or set()

        utils.compare_tracing_methods(
            module, tensor, fusible_ops=fusible_ops, fusion_blocklist=fusion_blocklist
        )
