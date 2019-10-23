from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests.utils import jitVsGlow


def test_constant_chunk_basic():
    """Test of prim::ConstantChunk node on glow"""

    def constant_chunk_basic(x, y):
        a = torch.chunk(x, 3, 1)  # shapes: [(10,4), (10,4), (10,3)]
        b = a[0]
        c = torch.chunk(y, 2, 1)  # shapes: [(10,4), (10, 3)]
        return b + a[1] + c[0], a[2] + c[1]

    x = torch.rand((10, 11))
    y = torch.rand((10, 7))

    jitVsGlow(constant_chunk_basic, x, y,
              expected_fused_ops={"prim::ConstantChunk"})
