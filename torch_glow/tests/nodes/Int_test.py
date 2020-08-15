from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestInt(unittest.TestCase):
    def test_Int(self):
        """Basic test of the PyTorch Int Node on Glow, along with constant
        propagation. Using int32 dtype, and aten::add."""

        def test_f(a):
            b = a.size(0)
            c = a.size(1)
            bt = torch.ops.prim.NumToTensor(b)
            ct = torch.ops.prim.NumToTensor(c)
            d = bt + ct
            d = d.to(torch.int32)
            i = torch.ops.aten.Int(d)
            res = torch.ops.prim.NumToTensor(i)
            return res

        x = torch.randn(2, 3, 4, dtype=torch.float32)
        jitVsGlow(test_f, x, expected_fused_ops={"aten::Int"}, use_script=True)

    def test_Int_mul_long(self):
        """Basic test of the PyTorch Int Node on Glow, along with constant
        propagation. Using int64 dtype, and aten::mul"""

        def test_f(a):
            b = a.size(0)
            c = a.size(1)
            bt = torch.ops.prim.NumToTensor(b)
            ct = torch.ops.prim.NumToTensor(c)
            d = bt * ct
            i = torch.ops.aten.Int(d)
            res = torch.ops.prim.NumToTensor(i)
            return res

        x = torch.randn(2, 3, 4, dtype=torch.float32)
        jitVsGlow(test_f, x, expected_fused_ops={"aten::Int"}, use_script=True)
