from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow
import pytest


# Need to be able to export lists from Glow fused nodes
@pytest.mark.skip(reason="not ready")
def test_size_basic():
    """Test of the PyTorch aten::size Node on Glow."""

    def test_f(a):
        b = a + a.size(0)
        return b

    x = torch.zeros([4], dtype=torch.int32)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::size"})
