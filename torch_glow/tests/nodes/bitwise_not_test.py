from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleBitwiseNotModule(torch.nn.Module):
    def __init__(self):
        super(SimpleBitwiseNotModule, self).__init__()

    def forward(self, a):
        b = torch.bitwise_not(a)
        return torch.bitwise_not(b)


class TestBitwiseNot(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", torch.tensor([-1, -2, 3], dtype=torch.int32)),
            ("basic_int64", torch.tensor([-1, -2, 3], dtype=torch.int64)),
            (
                "rand_int",
                torch.randint(-1000000000, 1000000000, (2, 3), dtype=torch.int64),
            ),
            ("bool_ts", torch.zeros((2, 2, 3), dtype=torch.bool)),
            ("bool_fs", torch.ones((2, 2, 3), dtype=torch.bool)),
            ("bool_tf", torch.tensor([False, True], dtype=torch.bool)),
        ]
    )
    def test_bitwise_not(self, _, x):
        """Tests of the PyTorch Bitwise Not Node on Glow."""
        utils.compare_tracing_methods(
            SimpleBitwiseNotModule(),
            x,
            fusible_ops={"aten::bitwise_not"},
        )
