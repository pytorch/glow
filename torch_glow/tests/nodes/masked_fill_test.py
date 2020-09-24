from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestMaskedFill(unittest.TestCase):
    def test_masked_fill_basic(self):
        """Test of the PyTorch aten::masked_fill op on Glow."""

        def masked_fill(a, mask):
            return torch.masked_fill(a + a, mask, 42.0)

        x = torch.randn([3])
        m = torch.tensor([True, False, True], dtype=torch.bool)

        jitVsGlow(masked_fill, x, m, expected_fused_ops={"aten::masked_fill"})

    def test_masked_fill_broadcasted(self):
        """Test of the PyTorch aten::masked_fill op on Glow with a
        broadcasted mask"""

        def masked_fill(a, mask):
            return torch.masked_fill(a + a, mask, 42.0)

        x = torch.randn([4, 1, 3])
        m = torch.tensor([True, False, True], dtype=torch.bool)

        jitVsGlow(masked_fill, x, m, expected_fused_ops={"aten::masked_fill"})

    def test_masked_fill_broadcasted_unit_dim(self):
        """Test of the PyTorch aten::masked_fill op on Glow with a
        broadcasted mask where the mask's size contains a leading 1"""

        def masked_fill(a, mask):
            return torch.masked_fill(a + a, mask, 42.0)

        x = torch.randn([4, 1, 3])
        m = torch.tensor([[True, False, True]], dtype=torch.bool)

        jitVsGlow(masked_fill, x, m, expected_fused_ops={"aten::masked_fill"})

    def test_masked_fill_broadcasted_multi_dim(self):
        """Test of the PyTorch aten::masked_fill op on Glow with a
        broadcasted mask where the mask's size has a non 1 lead dim"""

        def masked_fill(a, mask):
            return torch.masked_fill(a + a, mask, 42.0)

        x = torch.randn([2, 4, 3, 3])
        m = torch.tensor(
            [[[[True, False, True]]], [[[True, False, True]]]], dtype=torch.bool
        )

        jitVsGlow(masked_fill, x, m, expected_fused_ops={"aten::masked_fill"})

    def test_masked_fill_inplace(self):
        """Test of the PyTorch aten::masked_fill_ op on Glow"""

        def masked_fill(a, mask):
            b = a + a
            b.masked_fill_(mask, 42.0)
            return b

        x = torch.randn([3])
        m = torch.tensor([True, False, True], dtype=torch.bool)

        # Expect fuser to out-of-place the operator
        jitVsGlow(masked_fill, x, m, expected_fused_ops={"aten::masked_fill"})
