from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleLeakyReluModule(torch.nn.Module):
    def __init__(self, negative_slope=1e-2, inplace=False):
        super(SimpleLeakyReluModule, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, a):
        return torch.nn.functional.leaky_relu(
            a, negative_slope=self.negative_slope, inplace=self.inplace
        )


class TestLeakyRelu(utils.TorchGlowTestCase):
    def test_leaky_relu_basic(self):
        x = torch.randn(10)
        utils.compare_tracing_methods(
            SimpleLeakyReluModule(),
            x,
            fusible_ops={"aten::leaky_relu"},
        )

    def test_leaky_relu_3d(self):
        x = torch.randn(2, 3, 5)
        utils.compare_tracing_methods(
            SimpleLeakyReluModule(),
            x,
            fusible_ops={"aten::leaky_relu"},
        )

    def test_leaky_relu_inplace(self):
        x = torch.randn(10)
        utils.compare_tracing_methods(
            SimpleLeakyReluModule(inplace=True),
            x,
            fusible_ops={"aten::leaky_relu_"},
        )
