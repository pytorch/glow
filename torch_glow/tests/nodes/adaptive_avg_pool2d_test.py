from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F

from tests.utils import jitVsGlow
import unittest


class TestAdaptiveAvgPool2d(unittest.TestCase):
    def test_adaptive_avg_pool2d_basic(self):
        """Basic test of PyTorch adaptive_avg_pool2d Node."""

        def test_f(inputs):
            return F.adaptive_avg_pool2d(inputs, (5, 5))

        inputs = torch.randn(3, 6, 14, 14)

        jitVsGlow(test_f, inputs, expected_fused_ops={
                  "aten::adaptive_avg_pool2d"})

    def test_adaptive_avg_pool2d_nonsquare_inputs(self):
        """Test of PyTorch adaptive_avg_pool2d Node with non-square inputs."""

        def test_f(inputs):
            return F.adaptive_avg_pool2d(inputs, (3, 3))

        inputs = torch.randn(3, 6, 13, 14)

        jitVsGlow(test_f, inputs, expected_fused_ops={
                  "aten::adaptive_avg_pool2d"})

    def test_adaptive_avg_pool2d_nonsquare_outputs(self):
        """Test of PyTorch adaptive_avg_pool2d Node with non-square outputs."""

        def test_f(inputs):
            return F.adaptive_avg_pool2d(inputs, (5, 3))

        inputs = torch.randn(3, 6, 14, 14)

        jitVsGlow(test_f, inputs, expected_fused_ops={
                  "aten::adaptive_avg_pool2d"})
