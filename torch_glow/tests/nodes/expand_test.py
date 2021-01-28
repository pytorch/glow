from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleExpandModule(torch.nn.Module):
    def __init__(self, sizes):
        super(SimpleExpandModule, self).__init__()
        self.sizes = sizes

    def forward(self, a):
        return a.expand(*self.sizes)


class TestExpand(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleExpandModule((3, 4)), torch.tensor([[1], [2], [3]])),
            ("neg_sizes", SimpleExpandModule((-1, 4)), torch.tensor([[1], [2], [3]])),
        ]
    )
    def test_expand(self, _, module, a):
        """Test of the PyTorch Expand Node on Glow."""

        utils.compare_tracing_methods(module, a, fusible_ops={"aten::expand"})
