from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleRsubModel(torch.nn.Module):
    def __init__(self):
        super(SimpleRsubModel, self).__init__()

    def forward(self, tensor, other):
        if other.size() == torch.Size([]):
            return torch.rsub((tensor * tensor), other.item())
        else:
            third = torch.rsub(tensor, other)
            return torch.rsub(third, third)


class TestRsub(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleRsubModel(), torch.randn(4), torch.randn(4)),
            (
                "broadcast",
                SimpleRsubModel(),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            (
                "broadcast",
                SimpleRsubModel(),
                torch.randn(8, 3, 4, 2),
                torch.randn(1, 2),
            ),
            (
                "broadcast",
                SimpleRsubModel(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2),
            ),
            ("float", SimpleRsubModel(), torch.randn(4), torch.tensor(13.293)),
            ("int", SimpleRsubModel(), torch.randn(4), torch.tensor(4)),
        ]
    )
    def test_rsub(self, _, module, tensor, other):
        utils.compare_tracing_methods(module, tensor, other, fusible_ops={"aten::rsub"})
