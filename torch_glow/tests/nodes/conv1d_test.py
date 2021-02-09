from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from collections import namedtuple

import torch
import torch.nn.functional as F
from parameterized import parameterized
from tests import utils


class SimpleConv1dModule(torch.nn.Module):
    def __init__(self, stride=1, padding=0, dilation=1, groups=1):
        super(SimpleConv1dModule, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, inputs, filters, bias=None):
        conv = F.conv1d(
            inputs,
            filters,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return F.relu(conv)


class TestConv1d(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic",
                SimpleConv1dModule(padding=3),
                torch.randn(1, 4, 5),
                torch.randn(8, 4, 3),
            ),
            (
                "with_bias",
                SimpleConv1dModule(padding=1),
                torch.randn(1, 4, 5),
                torch.randn(8, 4, 3),
                torch.randn(8),
            ),
            (
                "dilation",
                SimpleConv1dModule(padding=1, dilation=2),
                torch.randn(1, 4, 5),
                torch.randn(8, 4, 3),
            ),
            (
                "stride",
                SimpleConv1dModule(stride=2),
                torch.randn(1, 4, 5),
                torch.randn(8, 4, 3),
            ),
        ]
    )
    def test_conv1d(self, _, module, inputs, filters, bias=None):
        utils.compare_tracing_methods(
            module, inputs, filters, fusible_ops={"aten::_convolution"}
        )

    def test_conv1d_param_sweep(self):
        """
        Test of the PyTorch conv1d Node sweeping through various parameters of the
        Node to test that they work correctly.
        """

        wOpts = [3, 4]
        padOpts = [0, 1]
        groupsOpts = [1, 2]
        strideOpts = [1, 2]

        Setting = namedtuple("Setting", ["w", "p", "g", "s"])

        settings = [
            Setting(w=w, p=p, g=g, s=s)
            for w in wOpts
            for p in padOpts
            for g in groupsOpts
            for s in strideOpts
        ]

        for setting in settings:

            inputs = torch.randn(2, 4, setting.w)
            filters = torch.randn(8, int(4 / setting.g), 3)

            utils.compare_tracing_methods(
                SimpleConv1dModule(
                    padding=setting.p, stride=setting.s, groups=setting.g
                ),
                inputs,
                filters,
                fusible_ops={"aten::_convolution"},
            )
