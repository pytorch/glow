from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleFmodModule(torch.nn.Module):
    def __init__(self):
        super(SimpleFmodModule, self).__init__()

    def forward(self, a, b):
        if b.size() == torch.Size([]):
            c = a.fmod(b.item())
        else:
            c = a.fmod(b)
        return c.fmod(1.0)


class TestFmod(unittest.TestCase):
    @parameterized.expand(
        [
            ("int_tensor", SimpleFmodModule(), torch.tensor([14]), torch.tensor([10])),
            ("float_tensor", SimpleFmodModule(), torch.randn(4), torch.tensor(0.3)),
            (
                "basic_tensor",
                SimpleFmodModule(),
                torch.tensor([7.5]),
                torch.tensor([2.4]),
            ),
            ("int_number", SimpleFmodModule(), torch.tensor([14]), torch.tensor(10)),
            ("basic", SimpleFmodModule(), torch.randn(4), torch.randn(4)),
            (
                "broadcast",
                SimpleFmodModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            (
                "broadcast",
                SimpleFmodModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(1, 2),
            ),
            (
                "positive_broadcast",
                SimpleFmodModule(),
                torch.Tensor(8, 3, 4, 2).random_(0, 5),
                torch.Tensor(1, 2).random_(1, 5),
            ),
            (
                "positive_broadcast",
                SimpleFmodModule(),
                torch.Tensor(4, 2).random_(0, 5),
                torch.Tensor(8, 3, 4, 2).random_(1, 5),
            ),
        ]
    )
    def test_fmod(self, _, module, a, b):
        utils.compare_tracing_methods(module, a, b, fusible_ops={"aten::fmod"})
