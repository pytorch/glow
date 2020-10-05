from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestMin(unittest.TestCase):
    def test_elementwise_min(self):
        """Test of the PyTorch min Node on Glow."""

        def test_f(a, b):
            return torch.min(a + a, b + b)

        jitVsGlow(
            test_f, torch.randn(7), torch.randn(7), expected_fused_ops={"aten::min"}
        )
