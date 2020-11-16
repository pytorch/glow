from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleMaxModule(torch.nn.Module):
    def __init__(self):
        super(SimpleMaxModule, self).__init__()

    def forward(self, a, b):
        return torch.max(a + a, b + b)


class MaxModule(torch.nn.Module):
    def __init__(self, axis, keepdim=False):
        super(MaxModule, self).__init__()
        self.axis = axis
        self.keepdim = False

    def forward(self, a):
        return torch.max(a + a, self.axis, keepdim=self.keepdim)


class TestMaxElementwise(unittest.TestCase):
    def test_elementwise_max(self):
        """Test of the PyTorch max Node on Glow."""

        utils.compare_tracing_methods(
            SimpleMaxModule(), torch.randn(4), torch.randn(4), fusible_ops={"aten::max"}
        )


class TestMax(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", MaxModule(0), torch.randn(2, 3)),
            ("keepdim", MaxModule(0, True), torch.randn(5, 3)),
            ("axis_1", MaxModule(1, True), torch.randn(4, 2)),
        ]
    )
    def test_max(self, _, module, tensor, skip_to_glow=False):
        utils.compare_tracing_methods(
            module,
            tensor,
            fusible_ops={"aten::max"},
            skip_to_glow=skip_to_glow,
        )
