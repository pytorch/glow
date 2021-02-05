from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleCumSumModule(torch.nn.Module):
    def __init__(self, dtype=None):
        super(SimpleCumSumModule, self).__init__()
        self.dtype = dtype

    def forward(self, a):
        return torch.cumsum(a, dim=0, dtype=self.dtype)


class TestCumSum(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleCumSumModule(), torch.randn(1)),
            ("basic", SimpleCumSumModule(), torch.randn(8)),
            ("2d", SimpleCumSumModule(), torch.randn(3, 4)),
            ("3d", SimpleCumSumModule(), torch.randn(4, 2, 3)),
            ("4d", SimpleCumSumModule(), torch.randn(3, 4, 2, 5)),
            ("dtype", SimpleCumSumModule(dtype=torch.int32), torch.randn(3, 4, 2, 5)),
            (
                "dtype",
                SimpleCumSumModule(dtype=torch.float),
                torch.arange(10, dtype=torch.int64),
            ),
        ]
    )
    def test_cumsum(self, _, module, a):
        utils.compare_tracing_methods(module, a, fusible_ops={"aten::cumsum"})
