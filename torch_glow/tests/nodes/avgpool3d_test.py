from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests import utils


class SimpleAvgPool3dModule(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SimpleAvgPool3dModule, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, inputs):
        return F.avg_pool3d(inputs, self.kernel_size, padding=self.padding)


class TestAvgPool3d(unittest.TestCase):
    def test_avg_pool3d_basic(self):
        """Basic test of the PyTorch avg_pool3d Node on Glow."""

        inputs = torch.randn(1, 4, 5, 5, 5)

        utils.compare_tracing_methods(
            SimpleAvgPool3dModule(3), inputs, fusible_ops={"aten::avg_pool3d"}
        )

    def test_avg_pool3d_with_args(self):
        """Test of the PyTorch avg_pool3d Node with arguments on Glow."""
        inputs = torch.randn(1, 4, 10, 10, 10)

        utils.compare_tracing_methods(
            SimpleAvgPool3dModule(3, (4, 7, 7)),
            inputs,
            fusible_ops={"aten::avg_pool3d"},
        )
