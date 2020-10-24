from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleConstantPadNDModule(torch.nn.Module):
    def __init__(self, pads, value):
        super(SimpleConstantPadNDModule, self).__init__()
        self.pads = pads
        self.value = value

    def forward(self, a):
        return torch.nn.functional.pad(a + a, self.pads, value=self.value)


class TestConstantPadND(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleConstantPadNDModule([10, 0], 0.0), torch.randn(6)),
            ("2d", SimpleConstantPadNDModule([3, 2], 2.0), torch.randn(3, 2)),
            ("3d", SimpleConstantPadNDModule([4, 2, 1, 5], 6.0), torch.randn(3, 2, 4)),
            (
                "value",
                SimpleConstantPadNDModule([4, 2, 1, 5, 3, 1], 6.0),
                torch.randn(3, 2, 4),
            ),
        ]
    )
    def test_constant_pad_nd(self, _, module, tensor):
        utils.compare_tracing_methods(
            module, tensor, fusible_ops={"aten::constant_pad_nd"}
        )
