from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests.utils import jitVsGlow


class TestGelu(unittest.TestCase):
    def test_gelu_basic(self):
        """Basic test of the PyTorch gelu Node on Glow."""

        def test_f(a):
            return F.gelu(a + a)

        for i in range(100):
            x = torch.randn(10)
            jitVsGlow(
                test_f,
                x,
                check_trace=False,
                atol=1e-3,
                expected_fused_ops={"aten::gelu"},
            )
