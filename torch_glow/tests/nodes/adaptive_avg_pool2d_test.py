import torch
import torch.nn.functional as F
import torch_glow

from tests.utils import jitVsGlow

# Basic test of the PyTorch adaptive_avg_pool2d Node on Glow.
def test_adaptive_avg_pool2d_basic():

  def adaptive_avg_pool2d_basic(inputs):
      return F.adaptive_avg_pool2d(inputs, (5, 5))

  inputs = torch.randn(3, 6, 14, 14)

  jitVsGlow(adaptive_avg_pool2d_basic, inputs)


# Test of the PyTorch adaptive_avg_pool2d Node with non-square inputs on Glow.
def test_adaptive_avg_pool2d_nonsquare_inputs():

  def adaptive_avg_pool2d_nonsquare_inputs(inputs):
      return F.adaptive_avg_pool2d(inputs, (3, 3))

  inputs = torch.randn(3, 6, 13, 14)

  jitVsGlow(adaptive_avg_pool2d_nonsquare_inputs, inputs)


# Test of the PyTorch adaptive_avg_pool2d Node with non-square outputs on Glow.
def test_adaptive_avg_pool2d_nonsquare_outputs():

  def adaptive_avg_pool2d_nonsquare_outputs(inputs):
      return F.adaptive_avg_pool2d(inputs, (5, 3))

  inputs = torch.randn(3, 6, 14, 14)

  jitVsGlow(adaptive_avg_pool2d_nonsquare_outputs, inputs)