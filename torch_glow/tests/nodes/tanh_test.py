from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestTanh(unittest.TestCase):
    def test_tanh(self):
        """Basic test of the PyTorch aten::tanh Node on Glow."""

        def test_f(a):
            return (a + a).tanh()

        x = torch.randn(4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::tanh"})

    def test_tanh_inplace(self):
        """Basic test of the PyTorch aten::tanh_ Node on Glow."""

        def test_f(a):
            return (a + a).tanh_()

        x = torch.randn(4)

        # Expect fuser to out-of-place the operator
        jitVsGlow(test_f, x, expected_fused_ops={"aten::tanh"})
