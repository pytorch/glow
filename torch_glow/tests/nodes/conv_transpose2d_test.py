from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from collections import namedtuple

import torch
import torch.nn.functional as F
from parameterized import parameterized
from tests import utils


class SimpleConvTranspose2dModule(torch.nn.Module):
    def __init__(self, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
        super(SimpleConvTranspose2dModule, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

    def forward(self, inputs, filters, bias=None):
        convTranspose = F.conv_transpose2d(
            inputs,
            filters,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )
        return F.relu(convTranspose)


class TestConvTranpose2d(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic",
                SimpleConvTranspose2dModule(padding=1),
                torch.randn(1, 4, 5, 5),
                torch.randn(4, 8, 3, 3),
            ),
            (
                "with_bias",
                SimpleConvTranspose2dModule(padding=1),
                torch.randn(1, 4, 5, 5),
                torch.randn(4, 8, 3, 3),
                torch.randn(4),
            ),
        ]
    )
    def test_convTranpose2d(self, _, module, inputs, filters, bias=None):
        """Basic test of the PyTorch conv3d Node on Glow."""

        utils.compare_tracing_methods(
            module, inputs, filters, fusible_ops={"aten::_convolution"}
        )
