from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_transpose_2d():
    """Test of PyTorch t (transpose) on Glow with 2d inputs."""

    def test_f(a):
        b = a + a
        return b.t()

    x = torch.randn(7, 4)

    jitVsGlow(test_f, x)


def test_transpose_1d():
    """Test of PyTorch t (transpose) on Glow with 1d inputs."""

    def test_f(a):
        b = a + a
        return b.t()

    x = torch.randn(7)

    jitVsGlow(test_f, x)


def test_transpose_inplace():
    """Test of PyTorch t_ (in place transpose) on Glow."""

    def test_f(a):
        b = a + a
        return b.t_()

    x = torch.randn(7, 4)

    jitVsGlow(test_f, x)
