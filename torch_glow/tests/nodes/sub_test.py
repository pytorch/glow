from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleSubtractModel(torch.nn.Module):
    def __init__(self):
        super(SimpleSubtractModel, self).__init__()

    def forward(self, a, b):
        if b.size() == torch.Size([]):
            return (a * a).sub(b.item())
        else:
            c = a.sub(b)
            return c.sub(c)


class TestSub(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleSubtractModel(), torch.randn(4), torch.randn(4)),
            (
                "broadcast_1",
                SimpleSubtractModel(),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            (
                "broadcast_2",
                SimpleSubtractModel(),
                torch.randn(8, 3, 4, 2),
                torch.randn(1, 2),
            ),
            (
                "broadcast_3",
                SimpleSubtractModel(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2),
            ),
            ("float", SimpleSubtractModel(), torch.randn(4), torch.tensor(3.9)),
            ("int", SimpleSubtractModel(), torch.randn(4), torch.tensor(20), True),
        ]
    )
    def test_subtract(self, _, module, tensor, other, skip_to_glow=False):
        utils.compare_tracing_methods(module, tensor, other, skip_to_glow=skip_to_glow)
