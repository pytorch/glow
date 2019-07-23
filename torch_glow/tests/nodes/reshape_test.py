import torch
import torch_glow

from tests.utils import *

# Basic test of the PyTorch div Node on Glow.
def test_reshape_basic():

  def reshape_basic(a, b):
      # type: (Tensor, Tuple[int, int]) -> Tensor
      c = a.reshape(b)
      return c

  x = torch.randn(2, 10)
  y = (4, 5)

  jitScriptVsGlow(reshape_basic, reshape_basic, x, y)
