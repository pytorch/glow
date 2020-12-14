from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleContiguousModel(torch.nn.Module):
    def __init__(self, memory_format=torch.contiguous_format):
        super(SimpleContiguousModel, self).__init__()
        self.memory_format = memory_format

    def forward(self, input):
        formatted = input.contiguous(memory_format=self.memory_format)
        return formatted + formatted


class TestContiguous(unittest.TestCase):
    def test_contiguous_basic(self):
        """Test of the PyTorch contiguous Node on Glow."""

        x = torch.randn(2, 2, 2)

        utils.compare_tracing_methods(
            SimpleContiguousModel(), x, fusible_ops={"aten::contiguous"}
        )

    def test_with_alternate_memory_format(self):

        x = torch.randn(3, 4, 5, 6)

        utils.compare_tracing_methods(
            SimpleContiguousModel(torch.channels_last),
            x,
            fusible_ops={"aten::contiguous"},
        )
