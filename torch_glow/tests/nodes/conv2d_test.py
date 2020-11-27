from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from collections import namedtuple

import torch
import torch.nn.functional as F
from parameterized import parameterized
from tests import utils


class SimpleConv2dModule(torch.nn.Module):
    def __init__(self, stride=1, padding=0, dilation=1, groups=1):
        super(SimpleConv2dModule, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, inputs, filters, bias=None):
        conv = F.conv2d(
            inputs,
            filters,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return F.relu(conv)


class TestConv2d(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic",
                SimpleConv2dModule(padding=1),
                torch.randn(1, 4, 5, 5),
                torch.randn(8, 4, 3, 3),
            ),
            (
                "with_bias",
                SimpleConv2dModule(padding=1),
                torch.randn(1, 4, 5, 5),
                torch.randn(8, 4, 3, 3),
                torch.randn(8),
            ),
            (
                "nonsquare_dilation",
                SimpleConv2dModule(padding=1, dilation=[1, 2]),
                torch.randn(1, 4, 5, 5),
                torch.randn(8, 4, 3, 3),
            ),
        ]
    )
    def test_conv2d(self, _, module, inputs, filters, bias=None):
        """Basic test of the PyTorch conv3d Node on Glow."""

        utils.compare_tracing_methods(
            module, inputs, filters, fusible_ops={"aten::_convolution"}
        )

    @unittest.skip(reason="not ready")
    def test_conv2d_param_sweep(self):
        """
        Test of the PyTorch conv2d Node sweeping through various parameters of the
        Node to test that they work correctly.
        """

        hwOpts = [3, 4]
        padOpts = [0, 1]
        groupsOpts = [1, 2]
        dilationOpts = [1, 2]
        strideOpts = [1, 2]

        Setting = namedtuple("Setting", ["h", "w", "p", "g", "d", "s"])

        settings = [
            Setting(h=h, w=w, p=p, g=g, d=d, s=s)
            for h in hwOpts
            for w in hwOpts
            for p in padOpts
            for g in groupsOpts
            for d in dilationOpts
            for s in strideOpts
        ]

        for setting in settings:

            def test_f(inputs, filters):
                conv = F.conv2d(inputs, filters, padding=setting.p, groups=setting.g)
                return F.relu(conv)

            inputs = torch.randn(2, 4, setting.h, setting.w)
            filters = torch.randn(8, 4 / setting.g, 3, 3)

            utils.compare_tracing_methods(
                test_f, inputs, filters, fusible_ops={"aten::_convolution"}
            )
