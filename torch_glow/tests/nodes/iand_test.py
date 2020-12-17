from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleIandModule(torch.nn.Module):
    def __init__(self):
        super(SimpleIandModule, self).__init__()

    def forward(self, a, b):
        a &= b
        return torch.logical_or(a, b)


class TestIand(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic",
                torch.tensor([True, True, False, False], dtype=torch.bool),
                torch.tensor([True, False, True, False], dtype=torch.bool),
            ),
            (
                "basic_3d",
                torch.zeros((3, 4, 5), dtype=torch.bool),
                torch.ones((3, 4, 5), dtype=torch.bool),
            ),
            (
                "broadcast_3d",
                torch.zeros((3, 4, 5), dtype=torch.bool),
                torch.ones((4, 5), dtype=torch.bool),
            ),
        ]
    )
    def test_iand(self, _, a, b, skip_to_glow=False):
        utils.compare_tracing_methods(
            SimpleIandModule(),
            a,
            b,
            fusible_ops={"aten::__iand__"},
            skip_to_glow=skip_to_glow,
        )
