from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestFloorDiv(unittest.TestCase):
    @unittest.skip(
        reason="Disabled while PyTorch floor_divide is fixed: github.com/pytorch/pytorch/issues/43874"
    )
    def test_floor_div_basic(self):
        """Basic test of the PyTorch div Node on Glow."""

        def test_f(a, b):
            return (a + a).floor_divide(1.9)

        x = torch.randn(4)
        y = torch.randn(4)
        print(x)
        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::floor_divide"})

    def test_floor_div_positive_basic(self):
        """Basic test of the PyTorch Floor div Node on Glow."""

        def test_f(a, b):
            return (a + a).floor_divide(b)

        x = torch.Tensor(4).random_(0, 5)
        y = torch.Tensor(4).random_(1, 5)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::floor_divide"})

    def test_floor_div_positive_float(self):
        """Test of the PyTorch aten::floor_divide Node with a float argument"""

        def test_f(a):
            return (a + a).floor_divide_(3.9)

        x = torch.Tensor(4).random_(0, 5)

        # Expect fuser to out-of-place the operator
        jitVsGlow(test_f, x, expected_fused_ops={"aten::floor_divide"})

    def test_floor_div_positive_broadcast_1(self):
        """Test of the PyTorch floor div Node on Glow with broadcasting."""

        def test_f(a, b):
            return (a + a).floor_divide(b)

        x = torch.Tensor(8, 3, 4, 2).random_(0, 5)
        y = torch.Tensor(4, 2).random_(1, 5)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::floor_divide"})

    def test_floor_div_positive_broadcast_2(self):
        """Test of the PyTorch floor div Node on Glow with broadcasting."""

        def test_f(a, b):
            return (a + a).floor_divide(b)

        x = torch.Tensor(8, 3, 4, 2).random_(0, 5)
        y = torch.Tensor(1, 2).random_(1, 5)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::floor_divide"})

    def test_floor_div_positive_broadcast_3(self):
        """Test of the PyTorch floor div Node on Glow with broadcasting."""

        def test_f(a, b):
            return (a + a).floor_divide(b)

        x = torch.Tensor(4, 2).random_(0, 5)
        y = torch.Tensor(8, 3, 4, 2).random_(1, 5)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::floor_divide"})
