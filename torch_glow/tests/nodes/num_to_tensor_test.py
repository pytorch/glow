from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestNumToTensor(unittest.TestCase):
    def test_NumToTensor_basic(self):
        """Basic test of the PyTorch NumToTensor Node on Glow."""

        def test_f(a):
            at0 = torch.ops.prim.NumToTensor(a.size(0))
            at1 = torch.ops.prim.NumToTensor(a.size(1))
            return torch.cat((at0.reshape(1), at1.reshape(1)))

        x = torch.randn(5, 6, 7)
        jitVsGlow(test_f, x, expected_fused_ops={"prim::NumToTensor"}, use_script=True)

    def test_NumToTensor_float(self):
        """Basic test of the PyTorch NumToTensor Node on Glow."""

        def test_f(a):
            at0 = torch.ops.prim.NumToTensor(a.size(0)).to(torch.float)
            # Const floating number is torch.float64 by-default
            # Therefore we need to convert it to float32 once NumToTensor is
            # used
            at1 = torch.ops.prim.NumToTensor(1.2).to(torch.float)
            return torch.cat((at0.reshape(1), at1.reshape(1)))

        x = torch.randn(5, 6, 7)
        jitVsGlow(test_f, x, expected_fused_ops={"prim::NumToTensor"}, use_script=True)
