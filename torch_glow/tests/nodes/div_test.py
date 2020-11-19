from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleDivModule(torch.nn.Module):
    def __init__(self):
        super(SimpleDivModule, self).__init__()

    def forward(self, a, b):
        if b.size() == torch.Size([]):
            return (a * a).div(b.item())
        else:
            c = a.div(b)
            return c.div(c)


class TestDiv(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleDivModule(), torch.randn(4), torch.randn(4)),
            (
                "broadcast",
                SimpleDivModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            (
                "broadcast",
                SimpleDivModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(1, 2),
            ),
            (
                "broadcast",
                SimpleDivModule(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2),
            ),
            ("float", SimpleDivModule(), torch.randn(4), torch.tensor(3.9)),
            ("int", SimpleDivModule(), torch.randn(4), torch.tensor(20), True),
        ]
    )
    def test_div_basic(self, _, module, a, b, skip_to_glow=False):
        utils.compare_tracing_methods(
            module, a, b, fusible_ops={"aten::div"}, skip_to_glow=skip_to_glow
        )
