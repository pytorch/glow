from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests.utils import jitVsGlow


class TestMaxPool2d(unittest.TestCase):
    def test_max_pool2d_basic(self):
        """Basic test of the PyTorch max_pool2d Node on Glow."""

        def test_f(inputs):
            return F.max_pool2d(inputs, 3)

        inputs = torch.randn(1, 4, 5, 5)

        jitVsGlow(test_f, inputs, expected_fused_ops={"aten::max_pool2d"})

    def test_max_pool2d_with_args(self):
        """Test of the PyTorch max_pool2d Node with arguments on Glow."""

        def test_f(inputs):
            return F.max_pool2d(inputs, padding=3, kernel_size=7)

        inputs = torch.randn(1, 4, 10, 10)

        jitVsGlow(test_f, inputs, expected_fused_ops={"aten::max_pool2d"})
