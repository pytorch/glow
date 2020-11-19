from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimplePermuteModule(torch.nn.Module):
    def __init__(self, *dimensions):
        super(SimplePermuteModule, self).__init__()
        self.dimensions = dimensions

    def forward(self, tensor):
        return tensor.permute(*self.dimensions)


class TestPermute(unittest.TestCase):
    def test_permute(self):
        """Basic test of the PyTorch aten::permute node on Glow."""

        utils.compare_tracing_methods(
            SimplePermuteModule(0, 2, 1),
            torch.randn(2, 3, 4),
            fusible_ops={"aten::permute"},
        )
