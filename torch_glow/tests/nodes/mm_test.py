from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleMmModule(torch.nn.Module):
    def __init__(self):
        super(SimpleMmModule, self).__init__()

    def forward(self, a, b, t):
        r = torch.mm(a, b)
        return r.mm(t)


class TestMm(unittest.TestCase):
    def test_mm_basic(self):
        """Test of the PyTorch mm Node on Glow."""

        x = torch.randn(2, 3)
        y = torch.randn(4, 3).t()
        t = torch.randn(4, 2)

        utils.compare_tracing_methods(
            SimpleMmModule(), x, y, t, fusible_ops={"aten::mm"}, skip_to_glow=True
        )
