import torch
import torch_glow

from tests.utils import jitVsGlow

# Basic test of the PyTorch div Node on Glow.
def test_div_basic():

  def div_basic(a, b):
        c = a.div(b)
        return c.div(c)

  x = torch.randn(4)
  y = torch.randn(4)

  jitVsGlow(div_basic, x, y)
