import torch
import torch_glow

from tests.utils import jitVsGlow

# Basic test of the PyTorch sub Node on Glow.
def test_sub_basic():

  def sub_basic(a, b):
        c = a.sub(b)
        return c.sub(c)

  x = torch.randn(4)
  y = torch.randn(4)

  jitVsGlow(sub_basic, x, y)