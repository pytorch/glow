from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleContiguousModel(torch.nn.Module):
    def forward(self, input):
        return input.contiguous()


class TestContiguous(unittest.TestCase):
    def test_contiguous_basic(self):
        """Test of the PyTorch contiguous Node on Glow."""

        x = torch.randn(2, 2, 2)

        utils.compare_tracing_methods(
            SimpleContiguousModel(), x, fusible_ops={"aten::contiguous"}
        )
