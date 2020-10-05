from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestMean(unittest.TestCase):
    def test_basic(self):
        """Test of the PyTorch mean Node on Glow."""

        def test_f(a, b):
            return torch.mean(a + b)

        jitVsGlow(
            test_f, torch.randn(7), torch.randn(7), expected_fused_ops={"aten::mean"}
        )

    def test_with_dims(self):
        """Test of the PyTorch mean node with dims on Glow. """

        def test_f(a, b):
            return torch.mean(a + b, (1, 2))

        x = torch.randn([1, 2, 3, 4])
        y = torch.randn([1, 2, 3, 4])
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::mean"})
