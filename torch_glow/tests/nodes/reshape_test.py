from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestReshape(unittest.TestCase):
    def test_reshape(self):
        """Test of the PyTorch reshape Node on Glow."""

        def test_f(a):
            b = a + a
            return b.reshape([2, -1])

        x = torch.rand(2, 3, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::reshape"})
