from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests.utils import jitVsGlow


class TestAvgPool3d(unittest.TestCase):
    def test_avg_pool3d_basic(self):
        """Basic test of the PyTorch avg_pool3d Node on Glow."""

        def test_f(inputs):
            return F.avg_pool3d(inputs, 3)

        inputs = torch.randn(1, 4, 5, 5, 5)

        jitVsGlow(test_f, inputs, expected_fused_ops={"aten::avg_pool3d"})

    def test_avg_pool3d_with_args(self):
        """Test of the PyTorch avg_pool3d Node with arguments on Glow."""

        def test_f(inputs):
            return F.avg_pool3d(inputs, padding=2, kernel_size=(4, 7, 7))

        inputs = torch.randn(1, 4, 10, 10, 10)

        jitVsGlow(test_f, inputs, expected_fused_ops={"aten::avg_pool3d"})
