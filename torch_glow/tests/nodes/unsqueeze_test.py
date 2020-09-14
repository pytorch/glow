from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestUnsqueeze(unittest.TestCase):
    def test_unsqueeze_dim_0(self):
        """Test of the PyTorch aten::unsqueeze Node on Glow."""

        def test_f(a):
            return torch.unsqueeze(a + a, 0)

        x = torch.randn(2, 3, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::unsqueeze"})

    def test_unsqueeze_dim_1(self):
        """Test of the PyTorch aten::unsqueeze Node on Glow."""

        def test_f(a):
            return torch.unsqueeze(a + a, 1)

        x = torch.randn(2, 3, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::unsqueeze"})

    def test_unsqueeze_dim_2(self):
        """Test of the PyTorch aten::unsqueeze Node on Glow."""

        def test_f(a):
            return torch.unsqueeze(a + a, 2)

        x = torch.randn(2, 3, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::unsqueeze"})

    def test_unsqueeze_dim_3(self):
        """Test of the PyTorch aten::unsqueeze Node on Glow."""

        def test_f(a):
            return torch.unsqueeze(a + a, 3)

        x = torch.randn(2, 3, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::unsqueeze"})

    def test_unsqueeze_negative_dim(self):
        """Test of the PyTorch aten::unsqueeze Node on Glow."""

        def test_f(a):
            return torch.unsqueeze(a + a, -1)

        x = torch.randn(2, 3, 4)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::unsqueeze"})

    def test_unsqueeze_inplace(self):
        """Test of the PyTorch aten::unsqueeze_ Node on Glow."""

        def test_f(a):
            b = a + a
            return b.unsqueeze_(-1)

        x = torch.randn(2, 3, 4)

        # Expect fuser to out-of-place the operator
        jitVsGlow(test_f, x, expected_fused_ops={"aten::unsqueeze"})
