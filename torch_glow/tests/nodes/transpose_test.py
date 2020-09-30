from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestTranspose(unittest.TestCase):
    def test_t_2d(self):
        """Test of PyTorch aten::t on Glow with 2d inputs."""

        def test_f(a):
            b = a + a
            return b.t()

        x = torch.randn(7, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::t"})

    def test_t_1d(self):
        """Test of PyTorch aten::t on Glow with 1d inputs."""

        def test_f(a):
            b = a + a
            return b.t()

        x = torch.randn(7)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::t"})

    def test_t_inplace(self):
        """Test of PyTorch aten::t_ (in place t) on Glow."""

        def test_f(a):
            b = a + a
            return b.t_()

        x = torch.randn(7, 4)

        # Expect fuser to out-of-place the operator
        jitVsGlow(test_f, x, expected_fused_ops={"aten::t"})

    def test_transpose(self):
        """Test of PyTorch aten::transpose on Glow."""

        def test_f(a):
            b = a + a
            return b.transpose(1, 2)

        x = torch.randn(2, 3, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::transpose"})

    def test_transpose_inplace(self):
        """Test of PyTorch aten::transpose on Glow."""

        def test_f(a):
            b = a + a
            return b.transpose_(1, 2)

        x = torch.randn(2, 3, 4)

        # Expect fuser to out-of-place the operator
        jitVsGlow(test_f, x, expected_fused_ops={"aten::transpose"})

    def test_transpose_neg_dim(self):
        """Test negative dimension index for PyTorch aten::transpose on Glow."""

        def test_f(a):
            b = a + a
            return b.transpose(-2, -1)

        def expected_f(a):
            b = a + a
            return b.transpose(1, 2)

        x = torch.randn(2, 3, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::transpose"})
        self.assertTrue(test_f(x).equal(expected_f(x)))

    def test_transpose_oob_neg_dim(self):
        """Test out of bounds negative dimension index for PyTorch aten::transpose on Glow."""

        def test_f(a):
            b = a + a
            return b.transpose(-2, -4)

        x = torch.randn(2, 3, 4)

        with self.assertRaises(IndexError):
            jitVsGlow(test_f, x, expected_fused_ops={"aten::transpose"})
