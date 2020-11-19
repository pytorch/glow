from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleNormModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(SimpleNormModule, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, tensor):
        return torch.norm(tensor, *self.args, **self.kwargs)


class TestNorm(unittest.TestCase):
    def test_norm_basic(self):
        """Basic test of the PyTorch norm Node on Glow."""

        utils.compare_tracing_methods(
            SimpleNormModule(dim=0, p=2),
            torch.arange(8, dtype=torch.float).reshape(2, 4),
            fusible_ops={"aten::norm"},
        )

    def test_norm_3d_inner_axis(self):
        """Basic test of the PyTorch norm Node on Glow."""

        utils.compare_tracing_methods(
            SimpleNormModule(dim=1),
            torch.arange(8, dtype=torch.float).reshape(2, 2, 2),
            fusible_ops={"aten::frobenius_norm"},
        )

    def test_norm_4d_outer_axis(self):
        """Basic test of the PyTorch norm Node on Glow."""

        utils.compare_tracing_methods(
            SimpleNormModule(dim=[3]),
            torch.arange(16, dtype=torch.float).reshape(2, 2, 2, 2),
            fusible_ops={"aten::frobenius_norm"},
        )
