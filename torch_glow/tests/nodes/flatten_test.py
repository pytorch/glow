from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleFlattenModule(torch.nn.Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super(SimpleFlattenModule, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return torch.flatten(input, start_dim=self.start_dim, end_dim=self.end_dim)


class TestFlatten(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleFlattenModule(), torch.randn(2, 3, 2, 5)),
            ("start_at_0", SimpleFlattenModule(0, 2), torch.randn(2, 3, 2, 5)),
            ("start_in_middle", SimpleFlattenModule(1, 2), torch.randn(2, 3, 2, 5)),
            ("negative_end_dim", SimpleFlattenModule(0, -2), torch.randn(2, 3, 2, 5)),
            ("same_dim", SimpleFlattenModule(2, 2), torch.randn(2, 3, 2, 5)),
            (
                "negative_start_dim",
                SimpleFlattenModule(-3, -1),
                torch.randn(2, 3, 2, 5),
            ),
        ]
    )
    def test_flatten(self, _, module, input):
        utils.compare_tracing_methods(module, input, fusible_ops={"aten::flatten"})
