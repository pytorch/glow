from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestStack(unittest.TestCase):
    def test_stack_basic(self):
        """Basic test of the PyTorch aten::stack Node on Glow."""

        def test_f(a, b):
            c = torch.stack((a, b), 0)
            d = torch.stack((c, c), 1)
            return torch.stack((d, d), 2)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        jitVsGlow(test_f, x, y, expected_fused_ops={"glow::fused_stack"})
