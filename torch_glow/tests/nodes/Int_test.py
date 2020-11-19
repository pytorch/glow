from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleIntModule(torch.nn.Module):
    def __init__(self, dtype):
        super(SimpleIntModule, self).__init__()
        # This has to be done in the init block, because control flow statements in the
        # forward method won't be fused during scripting.
        if dtype == torch.int32:
            self.forward = self._int32_forward
        else:
            self.forward = self._int64_forward

    def _int32_forward(self, a):
        b = a.size(0)
        c = a.size(1)
        bt = torch.ops.prim.NumToTensor(b)
        ct = torch.ops.prim.NumToTensor(c)
        d = bt + ct
        d = d.to(torch.int32)
        i = torch.ops.aten.Int(d)
        res = torch.ops.prim.NumToTensor(i)
        return res

    def _int64_forward(self, a):
        b = a.size(0)
        c = a.size(1)
        bt = torch.ops.prim.NumToTensor(b)
        ct = torch.ops.prim.NumToTensor(c)
        d = bt * ct
        i = torch.ops.aten.Int(d)
        res = torch.ops.prim.NumToTensor(i)
        return res


class TestInt(unittest.TestCase):
    def test_Int(self):
        """Basic test of the PyTorch Int Node on Glow, along with constant
        propagation. Using int32 dtype, and aten::add."""

        x = torch.randn(2, 3, 4, dtype=torch.float32)
        utils.compare_tracing_methods(
            SimpleIntModule(torch.int32), x, fusible_ops={"aten::Int"}, scripted=True
        )

    def test_Int_mul_long(self):
        """Basic test of the PyTorch Int Node on Glow, along with constant
        propagation. Using int64 dtype, and aten::mul"""

        x = torch.randn(2, 3, 4, dtype=torch.float32)
        utils.compare_tracing_methods(
            SimpleIntModule(torch.int64), x, fusible_ops={"aten::Int"}, scripted=True
        )
