from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleMaskedFillModule(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleMaskedFillModule, self).__init__()
        self.inplace = inplace

    def forward(self, tensor, mask):
        if self.inplace:
            other = tensor + tensor
            other.masked_fill_(mask, 42.0)
            return other
        else:
            return torch.masked_fill(tensor + tensor, mask, 42.0)


class TestMaskedFill(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic",
                SimpleMaskedFillModule(),
                torch.randn([3]),
                torch.tensor([True, False, True], dtype=torch.bool),
            ),
            (
                "broadcasted_unit_dim",
                SimpleMaskedFillModule(),
                torch.randn([4, 1, 3]),
                torch.tensor([True, True, True], dtype=torch.bool),
            ),
            (
                "broadcasted_multi_dim",
                SimpleMaskedFillModule(),
                torch.randn([2, 4, 3, 3]),
                torch.tensor(
                    [[[[True, False, True]]], [[[True, False, True]]]], dtype=torch.bool
                ),
            ),
            (
                "inplace",
                SimpleMaskedFillModule(True),
                torch.randn([3]),
                torch.tensor([True, False, True], dtype=torch.bool),
            ),
        ]
    )
    def test_masked_fill(self, _, module, tensor, mask):
        utils.compare_tracing_methods(
            module, tensor, mask, fusible_ops={"aten::masked_fill"}
        )
