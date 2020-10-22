from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests import utils


class SimpleAdapativeAvgPool2dModule(torch.nn.Module):
    def __init__(self, output_size):
        super(SimpleAdapativeAvgPool2dModule, self).__init__()
        self.output_size = output_size

    def forward(self, inputs):
        return F.adaptive_avg_pool2d(inputs, self.output_size)


class TestAdaptiveAvgPool2d(unittest.TestCase):
    def test_adaptive_avg_pool2d_basic(self):
        """Basic test of PyTorch adaptive_avg_pool2d Node."""
        inputs = torch.randn(3, 6, 14, 14)

        utils.compare_tracing_methods(
            SimpleAdapativeAvgPool2dModule((5, 5)),
            inputs,
            fusible_ops={"aten::adaptive_avg_pool2d"},
        )

    def test_adaptive_avg_pool2d_nonsquare_inputs(self):
        """Test of PyTorch adaptive_avg_pool2d Node with non-square inputs."""

        inputs = torch.randn(3, 6, 13, 14)

        utils.compare_tracing_methods(
            SimpleAdapativeAvgPool2dModule((3, 3)),
            inputs,
            fusible_ops={"aten::adaptive_avg_pool2d"},
        )

    def test_adaptive_avg_pool2d_nonsquare_outputs(self):
        """Test of PyTorch adaptive_avg_pool2d Node with non-square outputs."""

        inputs = torch.randn(3, 6, 14, 14)

        utils.compare_tracing_methods(
            SimpleAdapativeAvgPool2dModule((5, 3)),
            inputs,
            fusible_ops={"aten::adaptive_avg_pool2d"},
        )
