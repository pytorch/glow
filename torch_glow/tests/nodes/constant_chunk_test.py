from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleChunkModel(torch.nn.Module):
    def __init__(self, chunks, dimension):
        super(SimpleChunkModel, self).__init__()
        self.chunks = chunks
        self.dimension = dimension

    def forward(self, input):
        return torch.chunk(input + input, self.chunks, self.dimension)


class TestConstantChunk(unittest.TestCase):
    def test_constant_chunk_basic(self):
        """Test of prim::ConstantChunk node on glow"""

        x = torch.rand((10, 11))
        # shapes: [(10,4), (10,4), (10,3)]
        utils.compare_tracing_methods(
            SimpleChunkModel(3, 1),
            x,
            fusible_ops={"prim::ConstantChunk"},
            skip_to_glow=True,
        )

    def test_constant_chunk_negative_indices(self):
        """Test of prim::ConstantChunk node on glow"""

        x = torch.rand((10, 11))
        # shapes: [(4,11), (4,11), (2,11)]
        utils.compare_tracing_methods(
            SimpleChunkModel(3, -2),
            x,
            fusible_ops={"prim::ConstantChunk"},
            skip_to_glow=True,
        )
