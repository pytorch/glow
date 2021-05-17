from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F
from tests import utils


class SimpleAvgPool2dModule(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SimpleAvgPool2dModule, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def forward(self, inputs):
        return F.avg_pool2d(
            inputs + inputs, self.kernel_size, padding=self.padding, stride=self.stride
        )


class TestAvgPool2d(utils.TorchGlowTestCase):
    def test_avg_pool2d_basic(self):
        """Basic test of the PyTorch avg_pool2d Node on Glow."""

        inputs = torch.randn(1, 4, 5, 5)

        utils.run_comparison_tests(
            SimpleAvgPool2dModule(2),
            inputs,
            fusible_ops={"aten::avg_pool2d"},
        )

    def test_avg_pool2d_with_args(self):
        """Test of the PyTorch avg_pool2d Node with arguments on Glow."""

        inputs = torch.randn(1, 4, 10, 10)

        utils.run_comparison_tests(
            SimpleAvgPool2dModule(3, stride=7),
            inputs,
            fusible_ops={"aten::avg_pool2d"},
            fp16vfp16_atol=1e-3,
        )
