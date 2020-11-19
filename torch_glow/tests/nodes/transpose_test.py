from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleTransposeModel(torch.nn.Module):
    def __init__(self, dim0=None, dim1=None, inplace=False):
        super(SimpleTransposeModel, self).__init__()
        self.dims = (dim0, dim1) if dim0 and dim1 else None
        self.inplace = inplace

    def forward(self, tensor):
        t = tensor + tensor
        if self.dims:
            return t.transpose_(*self.dims) if self.inplace else t.transpose(*self.dims)
        else:
            return t.t_() if self.inplace else t.t()


class TestTranspose(unittest.TestCase):
    @parameterized.expand(
        [
            ("2d", SimpleTransposeModel(), torch.randn(7, 4)),
            ("1d", SimpleTransposeModel(), torch.randn(7)),
            ("inplace", SimpleTransposeModel(inplace=True), torch.randn(7, 4)),
        ]
    )
    def test_t(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::t"})

    @parameterized.expand(
        [
            ("simple", SimpleTransposeModel(1, 2), torch.randn(2, 3, 4)),
            ("inplace", SimpleTransposeModel(1, 2, inplace=True), torch.randn(2, 3, 4)),
            ("neg_dim", SimpleTransposeModel(-2, -1), torch.randn(2, 3, 4)),
        ]
    )
    def test_transpose(self, _, module, tensor, reference=None):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::transpose"})

    @parameterized.expand(
        [("oob_neg_dim", SimpleTransposeModel(-2, -4), torch.randn(2, 3, 4))]
    )
    def test_transpose_failure(self, _, module, tensor):
        with self.assertRaises(IndexError):
            utils.compare_tracing_methods(
                module, tensor, fusible_ops={"aten::transpose"}
            )
