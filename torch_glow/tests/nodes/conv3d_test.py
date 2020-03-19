from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F
from collections import namedtuple

from tests.utils import jitVsGlow
import unittest


class TestConv3d(unittest.TestCase):
    def test_conv3d_basic(self):
        """Basic test of the PyTorch conv3d Node on Glow."""

        def test_f(inputs, filters):
            conv = F.conv3d(inputs, filters, padding=1)
            return F.relu(conv)

        inputs = torch.randn(1, 4, 5, 5, 3)
        filters = torch.randn(8, 4, 3, 3, 3)

        jitVsGlow(test_f, inputs, filters,
                  expected_fused_ops={"aten::_convolution"})

    def test_conv3d_with_bias(self):
        """Test of the PyTorch conv3d Node with a provided bias tensor."""

        def test_f(inputs, filters, bias):
            conv = F.conv3d(inputs, filters, bias)
            return F.relu(conv)

        inputs = torch.randn(1, 4, 5, 5, 3)
        filters = torch.randn(8, 4, 3, 3, 3)
        bias = torch.randn(8)

        jitVsGlow(test_f, inputs, filters, bias,
                  expected_fused_ops={"aten::_convolution"})

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

            def test_f(inputs, filters):
                conv = F.conv3d(
                    inputs,
                    filters,
                    padding=setting.p,
                    stride=setting.s,
                    groups=setting.g)
                return F.relu(conv)

            inputs = torch.randn(2, 4, setting.t, setting.h, setting.w)
            filters = torch.randn(8, int(4 / setting.g), 3, 3, 3)

            jitVsGlow(test_f, inputs, filters,
                      expected_fused_ops={"aten::_convolution"})
