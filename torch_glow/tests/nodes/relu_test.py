import torch
import torch.nn.functional as F
import torch_glow

from tests.utils import jitVsGlow

# Basic test of the PyTorch relu Node on Glow.
def test_relu_basic():

  def relu_basic(a):
        b = F.relu(a)
        return F.relu(b)

  x = torch.randn(4)
  # make sure we have at least one negative
  x[0] = -2.0

  jitVsGlow(relu_basic, x)