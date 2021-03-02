from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F
from tests import utils


class SimpleAvgPool1dModule(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SimpleAvgPool1dModule, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def forward(self, inputs):
        return F.avg_pool1d(
            inputs, self.kernel_size, padding=self.padding, stride=self.stride
        )


class TestAvgPool1d(utils.TorchGlowTestCase):
    def test_avg_pool1d_basic(self):
        """Basic test of the PyTorch avg_pool1d Node on Glow."""
        inputs = torch.randn(1, 4, 9)

        utils.compare_tracing_methods(
            SimpleAvgPool1dModule(3), inputs, fusible_ops={"aten::avg_pool1d"}
        )

    def test_avg_pool1d_with_args(self):
        """Test of the PyTorch avg_pool1d Node with arguments on Glow."""

        inputs = torch.randn(1, 4, 10)

        utils.compare_tracing_methods(
            SimpleAvgPool1dModule(3, stride=7), inputs, fusible_ops={"aten::avg_pool1d"}
        )
