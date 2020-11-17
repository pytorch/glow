from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleAllModule(torch.nn.Module):
    def __init__(self):
        super(SimpleAllModule, self).__init__()

    def forward(self, a):
        b = torch.logical_not(a)
        return torch.all(b)


class TestAllBasic(unittest.TestCase):
    def test_all_basic(self):
        a = torch.rand(2, 3, 4).bool()
        utils.compare_tracing_methods(SimpleAllModule(), a, fusible_ops={"aten::all"})


class KeepDimAllModule(torch.nn.Module):
    def __init__(self, axis, keepdim):
        super(KeepDimAllModule, self).__init__()
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, a):
        b = torch.logical_not(a)
        return torch.all(b, self.axis, keepdim=self.keepdim)


class TestAllKeepDim(unittest.TestCase):
    @parameterized.expand(
        [
            ("keepdim", KeepDimAllModule(0, True), torch.rand(2, 3, 4).bool()),
            ("axis_1", KeepDimAllModule(1, False), torch.rand(4, 3, 4).bool()),
            (
                "axis_2_keepdim",
                KeepDimAllModule(2, True),
                torch.rand(5, 2, 4).bool(),
            ),
            (
                "neg_axis",
                KeepDimAllModule(-2, False),
                torch.zeros(3, 1, 2).bool(),
            ),
            (
                "neg_axis_keepdim",
                KeepDimAllModule(-2, True),
                torch.zeros(3, 5, 2).bool(),
            ),
        ]
    )
    def test_all(self, _, module, a):
        utils.compare_tracing_methods(module, a, fusible_ops={"aten::all"})
