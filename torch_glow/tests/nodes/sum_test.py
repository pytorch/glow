from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleSumModule(torch.nn.Module):
    def __init__(self, dtype=None):
        super(SimpleSumModule, self).__init__()
        self.dtype = dtype

    def forward(self, a):
        b = a + a
        return torch.sum(b, dtype=self.dtype)


class KeepdimSumModule(torch.nn.Module):
    def __init__(self, axis, keepdim, dtype=None):
        super(KeepdimSumModule, self).__init__()
        self.axis = axis
        self.keepdim = keepdim
        self.dtype = dtype

    def forward(self, a):
        b = a + a
        return torch.sum(b, self.axis, keepdim=self.keepdim, dtype=self.dtype)


class TestSumBasic(unittest.TestCase):
    def test_sum_basic(self):
        a = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(SimpleSumModule(), a, fusible_ops={"aten::sum"})


class TestSumKeepdim(unittest.TestCase):
    @parameterized.expand(
        [
            ("keepdim", KeepdimSumModule(0, True), torch.randn(2, 3, 4)),
            ("axis_1", KeepdimSumModule(1, False), torch.randn(4, 3, 4)),
            (
                "axis_2_keepdim_f16",
                KeepdimSumModule(2, True, torch.float16),
                torch.randn(5, 2, 4),
            ),
            (
                "axis_1_f16",
                KeepdimSumModule(1, False, torch.float16),
                torch.randn(3, 1, 2),
            ),
            (
                "neg_axis_f16",
                KeepdimSumModule(-2, False, torch.float16),
                torch.randn(3, 1, 2),
            ),
            (
                "neg_axis_keepdim",
                KeepdimSumModule(-2, True),
                torch.randn(3, 1, 2),
            ),
        ]
    )
    def test_sum(self, _, module, a):
        utils.compare_tracing_methods(module, a, fusible_ops={"aten::sum"})
