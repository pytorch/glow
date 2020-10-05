from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestPow(unittest.TestCase):
    def test_pow_basic(self):
        """Test of the PyTorch pow Node on Glow."""

        def pow_basic(a):
            b = torch.pow(a, 2.3)
            return torch.pow(b, 3.4)

        x = torch.rand(4) + 5

        jitVsGlow(pow_basic, x, expected_fused_ops={"aten::pow"})
