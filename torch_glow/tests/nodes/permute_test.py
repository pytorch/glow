from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestPermute(unittest.TestCase):
    def test_permute(self):
        """Basic test of the PyTorch aten::permute node on Glow."""

        def test_f(a):
            b = a.permute(0, 2, 1)
            return b

        x = torch.randn(2, 3, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::permute"})
