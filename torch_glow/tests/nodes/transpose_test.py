from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_t_2d():
    """Test of PyTorch aten::t on Glow with 2d inputs."""

    def test_f(a):
        b = a + a
        return b.t()

    x = torch.randn(7, 4)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::t"})


def test_t_1d():
    """Test of PyTorch aten::t on Glow with 1d inputs."""

    def test_f(a):
        b = a + a
        return b.t()

    x = torch.randn(7)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::t"})


def test_t_inplace():
    """Test of PyTorch aten::t_ (in place t) on Glow."""

    def test_f(a):
        b = a + a
        return b.t_()

    x = torch.randn(7, 4)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::t_"})


def test_transpose():
    """Test of PyTorch aten::transpose on Glow."""

    def test_f(a):
        b = a + a
        return b.transpose(1, 2)

    x = torch.randn(2, 3, 4)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::transpose"})


def test_transpose_inplace():
    """Test of PyTorch aten::transpose on Glow."""

    def test_f(a):
        b = a + a
        return b.transpose_(1, 2)

    x = torch.randn(2, 3, 4)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::transpose_"})
