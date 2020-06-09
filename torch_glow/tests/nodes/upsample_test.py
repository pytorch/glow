from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch_glow
from tests.utils import jitVsGlow


class TestUpsample(unittest.TestCase):
    def test_upsample3d_2x_size(self):
        """Test of the PyTorch upsample Node on Glow."""

        def test_f(a):
            U = torch.nn.Upsample(size=(8, 10, 12))
            return U(a)

        x = torch.rand(2, 3, 4, 5, 6)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::upsample_nearest3d"})

    def test_upsample3d_2x_scale_factor(self):
        """Test of the PyTorch upsample Node on Glow."""

        def test_f(a):
            U = torch.nn.Upsample(scale_factor=(2, 2, 2))
            return U(a)

        x = torch.rand(2, 3, 4, 5, 6)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::upsample_nearest3d"})

    def test_upsample3d_2x_single_scale_factor(self):
        """Test of the PyTorch upsample Node on Glow."""

        def test_f(a):
            U = torch.nn.Upsample(scale_factor=2)
            return U(a)

        x = torch.rand(2, 3, 4, 5, 6)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::upsample_nearest3d"})

    def test_upsample3d_not_2x_single_scale_factor(self):
        """Test of the PyTorch upsample Node on Glow."""

        def test_f(a):
            U = torch.nn.Upsample(scale_factor=5)
            return U(a)

        x = torch.rand(2, 3, 4, 5, 6)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::upsample_nearest3d"})

    @unittest.skip(
        reason="breaks because of constants not being differentiated in CachingGraphRunner"
    )
    def test_upsample3d_not_2x_scale_factor(self):
        """Test of the PyTorch upsample Node on Glow."""

        def test_f(a):
            U = torch.nn.Upsample(scale_factor=(1, 2, 3))
            return U(a)

        x = torch.rand(2, 2, 2, 2, 2)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::upsample_nearest3d"})

    def test_upsample3d_not_2x_size(self):
        """Test of the PyTorch upsample Node on Glow."""

        def test_f(a):
            U = torch.nn.Upsample(size=(10, 12, 13))
            return U(a)

        x = torch.rand(2, 3, 4, 5, 6)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::upsample_nearest3d"})
