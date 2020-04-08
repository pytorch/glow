from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests.utils import jitVsGlow
import unittest


class TestView(unittest.TestCase):
    def test_view(self):
        """Test of the PyTorch reshape Node on Glow."""

        def test_f(a):
            b = a + a
            return b.view([2, -1])

        x = torch.rand(2, 3, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::view"})
