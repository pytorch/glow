from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from collections import namedtuple

import torch
import torch.nn.functional as F
from parameterized import parameterized
from tests import utils


class SimpleConv3dModule(torch.nn.Module):
    def __init__(self, stride=1, padding=0, dilation=1, groups=1):
        super(SimpleConv3dModule, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, inputs, filters, bias=None):
        conv = F.conv3d(
            inputs,
            filters,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return F.relu(conv)


class TestConv3d(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic",
                SimpleConv3dModule(padding=1),
                torch.randn(1, 4, 5, 5, 3),
                torch.randn(8, 4, 3, 3, 3),
            ),
            (
                "with_bias",
                SimpleConv3dModule(padding=1),
                torch.randn(1, 4, 5, 5, 3),
                torch.randn(8, 4, 3, 3, 3),
                torch.randn(8),
            ),
        ]
    )
    def test_conv3d(self, _, module, inputs, filters, bias=None):
        """Basic test of the PyTorch conv3d Node on Glow."""

        utils.compare_tracing_methods(
            module, inputs, filters, fusible_ops={"aten::_convolution"}
        )

    def test_conv3d_param_sweep(self):
        """
        Test of the PyTorch conv3d Node sweeping through various parameters of the
        Node to test that they work correctly.
        """

        thwOpts = [3, 4]
        padOpts = [0, 1]
        groupsOpts = [1, 2]
        strideOpts = [1, 2]

        Setting = namedtuple("Setting", ["t", "h", "w", "p", "g", "s"])

        settings = [
            Setting(t=t, h=h, w=w, p=p, g=g, s=s)
            for t in thwOpts
            for h in thwOpts
            for w in thwOpts
            for p in padOpts
            for g in groupsOpts
            for s in strideOpts
        ]

        for setting in settings:

            inputs = torch.randn(2, 4, setting.t, setting.h, setting.w)
            filters = torch.randn(8, int(4 / setting.g), 3, 3, 3)

            utils.compare_tracing_methods(
                SimpleConv3dModule(
                    padding=setting.p, stride=setting.s, groups=setting.g
                ),
                inputs,
                filters,
                fusible_ops={"aten::_convolution"},
            )
