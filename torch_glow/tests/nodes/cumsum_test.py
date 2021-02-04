from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleCumSumModule(torch.nn.Module):
    def __init__(self):
        super(SimpleCumSumModule, self).__init__()

    def forward(self, tensor):
        # TODO remove default of 0 when axis/dimension to sum is supported
        return torch.cumsum(tensor, dim=0)


class TestCumSum(unittest.TestCase):
    @parameterized.expand(
        [
            ("1", torch.randn(1)),
            ("2", torch.randn(2)),
            ("20", torch.randn(20)),
            # TODO add these tests when multi-dimension is supported
            # ("3x3", torch.randn(3, 3)),
            # ("5x4", torch.randn(5, 4)),
            # ("3x3x3", torch.randn(3, 4, 5)),
            # ("3x4x5", torch.randn(3, 4, 5)),
            # ("4x4x4x4", torch.randn(6, 5, 4, 3)),
            # ("6x5x4x3", torch.randn(6, 5, 4, 3)),
        ]
    )
    def test_cumsum(self, _, tensor):
        utils.compare_tracing_methods(
            SimpleCumSumModule(), tensor, fusible_ops={"aten::cumsum"}
        )
