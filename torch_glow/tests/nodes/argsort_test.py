from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleArgSortModule(torch.nn.Module):
    def __init__(self, descending=True):
        super(SimpleArgSortModule, self).__init__()
        self.descending = descending

    def forward(self, inputs):
        # Only last dim is currently supported
        return torch.argsort(inputs, dim=-1, descending=self.descending)


class TestArgSort(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "desc",
                SimpleArgSortModule(),
                torch.randn(4),
            ),
            (
                "asc",
                SimpleArgSortModule(descending=False),
                torch.randn(4),
            ),
            (
                "2d_desc",
                SimpleArgSortModule(),
                torch.randn(4, 3),
            ),
            (
                "3d_asc",
                SimpleArgSortModule(descending=False),
                torch.randn(6, 4, 5),
            ),
            (
                "4d_desc",
                SimpleArgSortModule(),
                torch.randn(4, 7, 7, 3),
            ),
        ]
    )
    def test_argsort(self, _, module, a):
        utils.compare_tracing_methods(module, a, fusible_ops={"aten::argsort"})
