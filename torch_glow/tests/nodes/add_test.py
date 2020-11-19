from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleAddModule(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleAddModule, self).__init__()
        self.inplace = inplace

    def forward(self, a, b):
        if b.size() == torch.Size([]):
            return (a * a).add(b.item())
        if self.inplace:
            c = a.add_(b)
            return c.add_(c)
        else:
            c = a.add(b)
            return c.add(c)


class TestAdd(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleAddModule(), torch.randn(4), torch.randn(4)),
            ("inplace", SimpleAddModule(True), torch.randn(4), torch.randn(4)),
            (
                "broadcast",
                SimpleAddModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            (
                "broadcast",
                SimpleAddModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(1, 2),
            ),
            (
                "broadcast",
                SimpleAddModule(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2),
            ),
            ("float", SimpleAddModule(), torch.randn(4), torch.tensor(1.2345)),
            ("int", SimpleAddModule(), torch.randn(4), torch.tensor(42), True),
        ]
    )
    def test_add(self, _, module, a, b, skip_to_glow=False):
        utils.compare_tracing_methods(
            module,
            a,
            b,
            skip_to_glow=skip_to_glow,
            fusible_ops={"aten::add_"} if module.inplace else {"aten::add"},
        )
