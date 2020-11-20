from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleQuantizedAvgPool3DModule(torch.nn.Module):
    def __init__(self, scale, zero_point, dtype, kernel_size):
        super(SimpleQuantizedAvgPool3DModule, self).__init__()
        self.quantize = torch.nn.quantized.Quantize(
            scale=scale, zero_point=zero_point, dtype=dtype
        )
        self.average_pool = torch.nn.AvgPool3d(kernel_size)

    def forward(self, inputs):
        return torch.nn.quantized.DeQuantize()(self.average_pool(self.quantize(inputs)))


class TestQuantizedAvgPool3D(unittest.TestCase):
    def test_quantized_avgpool3d(self):
        """Basic test of the PyTorch quantized::avg_pool2d Node on Glow."""

        utils.compare_tracing_methods(
            SimpleQuantizedAvgPool3DModule(1.0 / 128, 3, torch.quint8, 3),
            torch.randn(1, 2, 4, 5, 5),
            fusible_ops={
                "aten::avg_pool3d",
                "aten::quantize_per_tensor",
                "aten::dequantize",
            },
        )

    def test_quantized_avgpool_cut_q_dq(self):
        """Basic test of the PyTorch quantized::avg_pool2d Node on Glow, with quantize and dequantize excluded. """

        utils.compare_tracing_methods(
            SimpleQuantizedAvgPool3DModule(1.0 / 128, 3, torch.quint8, 3),
            torch.randn(1, 2, 4, 5, 5),
            fusible_ops={"aten::avg_pool3d"},
            fusion_blocklist=["aten::quantize_per_tensor", "aten::dequantize"],
        )
