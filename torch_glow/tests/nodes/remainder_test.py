from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleRemainderModel(torch.nn.Module):
    def __init__(self, other):
        super(SimpleRemainderModel, self).__init__()
        self.other = other

    def forward(self, tensor):
        return torch.remainder(tensor + tensor, self.other)


class TestRemainder(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleRemainderModel(torch.randn(2, 3, 4)), torch.randn(2, 3, 4)),
            (
                "broadcast",
                SimpleRemainderModel(torch.randn(3, 4)),
                torch.randn(2, 3, 4),
            ),
            ("scalar", SimpleRemainderModel(2), torch.randn(2, 3, 4)),
        ]
    )
    def test_remainder(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::remainder"})
