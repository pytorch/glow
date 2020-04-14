from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests.utils import jitVsGlow
import unittest


class TestNumToTensor(unittest.TestCase):
    def test_view(self):
        """Test of the PyTorch NumToTensor on Glow."""

        def test_f(a):
            a = a.size(0)
            b = a
            return b

        x = torch.randn(4)

        jitVsGlow(test_f, x, expected_fused_ops={"prim::NumToTensor"})
