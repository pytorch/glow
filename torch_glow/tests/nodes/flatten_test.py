from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestFlatten(unittest.TestCase):
    def test_flatten_basic(self):
        """Test of the PyTorch flatten Node on Glow."""

        def test_f(a):
            return torch.flatten(a)

        x = torch.randn(2, 3, 2, 5)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::flatten"})

    def test_flatten_start_at_0(self):
        """Test of the PyTorch flatten Node on Glow."""

        def test_f(a):
            return torch.flatten(a, start_dim=0, end_dim=2)

        x = torch.randn(2, 3, 2, 5)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::flatten"})

    def test_flatten_start_in_middle(self):
        """Test of the PyTorch flatten Node on Glow."""

        def test_f(a):
            return torch.flatten(a, start_dim=1, end_dim=2)

        x = torch.randn(2, 3, 2, 5)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::flatten"})

    def test_flatten_negative_end_dim(self):
        """Test of the PyTorch flatten Node on Glow."""

        def test_f(a):
            return torch.flatten(a, start_dim=0, end_dim=-2)

        x = torch.randn(2, 3, 2, 5)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::flatten"})

    def test_flatten_same_dim(self):
        """Test of the PyTorch flatten Node on Glow."""

        def test_f(a):
            return torch.flatten(a, start_dim=2, end_dim=2)

        x = torch.randn(2, 3, 2, 5)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::flatten"})

    def test_flatten_negative_start_dim(self):
        """Test of the PyTorch flatten Node on Glow."""

        def test_f(a):
            return torch.flatten(a, start_dim=-3, end_dim=-1)

        x = torch.randn(2, 3, 2, 5)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::flatten"})
