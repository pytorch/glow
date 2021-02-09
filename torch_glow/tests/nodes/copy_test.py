from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class CopyModule(torch.nn.Module):
    def __init__(self):
        super(CopyModule, self).__init__()

    def forward(self, a, b):
        a.copy_(b)
        return a + a


class TestCopy(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic",
                CopyModule(),
                torch.zeros(3).float(),
                torch.randn(3).float(),
            ),
            (
                "2d",
                CopyModule(),
                torch.zeros(2, 3),
                torch.randn(2, 3),
            ),
            (
                "3d",
                CopyModule(),
                torch.zeros(2, 3, 4),
                torch.randn(2, 3, 4),
            ),
            (
                "broadcast",
                CopyModule(),
                torch.zeros(2, 3, 4),
                torch.randn(3, 4),
            ),
            (
                "broadcast",
                CopyModule(),
                torch.zeros(2, 3, 4),
                torch.randn(1, 1, 4),
            ),
            (
                "broadcast",
                CopyModule(),
                torch.zeros(1, 2, 3, 4),
                torch.randn(1, 1, 3, 1),
            ),
            (
                "broadcast",
                CopyModule(),
                torch.zeros(2, 3, 4),
                torch.randn(4),
            ),
            (
                "dtype",
                CopyModule(),
                torch.zeros(2, 3, 4, dtype=torch.int32),
                torch.randn(4, dtype=torch.float32),
            ),
            (
                "dtype",
                CopyModule(),
                torch.zeros(2, 3, 4, dtype=torch.float32),
                torch.randn(4, dtype=torch.float16),
            ),
        ]
    )
    def test_copy(self, _, module, a, b):
        utils.compare_tracing_methods(module, a, b, fusible_ops={"aten::copy_"})
