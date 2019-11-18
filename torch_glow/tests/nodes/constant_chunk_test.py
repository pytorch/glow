from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests.utils import jitVsGlow


def test_constant_chunk_basic():
    """Test of prim::ConstantChunk node on glow"""

    def test_f(x):
        return torch.chunk(x + x, 3, 1)  # shapes: [(10,4), (10,4), (10,3)]

    x = torch.rand((10, 11))

    jitVsGlow(test_f, x,
              expected_fused_ops={"prim::ConstantChunk"})


def test_constant_chunk_negative_indices():
    """Test of prim::ConstantChunk node on glow"""

    def test_f(x):
        return torch.chunk(x + x, 3, -2)  # shapes: [(4,11), (4,11), (2,11)]

    x = torch.rand((10, 11))

    jitVsGlow(test_f, x,
              expected_fused_ops={"prim::ConstantChunk"})
