import torch
import torch_glow

from tests.utils import jitVsGlow

# Basic test of the PyTorch mul Node on Glow.


def test_mul_basic():
    def mul_basic(a, b):
        c = a.mul(b)
        return c.mul(c)

    x = torch.randn(4)
    y = torch.randn(4)

    jitVsGlow(mul_basic, x, y)
