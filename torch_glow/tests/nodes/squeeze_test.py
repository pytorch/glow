from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestSqueeze(unittest.TestCase):
    def test_squeeze_basic(self):
        """Test of the PyTorch aten::squeeze Node on Glow."""

        def test_f(a):
            return torch.squeeze(a + a)

        x = torch.randn(1, 3, 1, 2, 5, 1)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::squeeze"})

    def test_squeeze_with_dim(self):
        """Test of the PyTorch aten::squeeze Node on Glow."""

        def test_f(a):
            return torch.squeeze(a + a, 2)

        x = torch.randn(1, 3, 1, 2, 5, 1)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::squeeze"})

    def test_squeeze_with_negative_dim(self):
        """Test of the PyTorch aten::squeeze Node on Glow."""

        def test_f(a):
            return torch.squeeze(a + a, -1)

        x = torch.randn(1, 3, 1, 2, 5, 1)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::squeeze"})

    def test_squeeze_inplace(self):
        """Test of the PyTorch aten::squeeze_ Node on Glow."""

        def test_f(a):
            b = a + a
            return b.squeeze_()

        x = torch.randn(1, 3, 1, 2, 5, 1)

        # Expect fuser to out-of-place the operator
        jitVsGlow(test_f, x, expected_fused_ops={"aten::squeeze"})
