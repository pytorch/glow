import torch
import torch.nn.functional as F
import torch_glow

from tests.utils import jitVsGlow

# Basic test of the PyTorch batchnorm Node on Glow.
def test_batchnorm_basic():

  def batchnorm_basic(inputs, running_mean, running_var):
    return F.batch_norm(inputs, running_mean, running_var)

  inputs = torch.randn(1, 4, 5, 5)
  running_mean = torch.rand(4)
  running_var = torch.rand(4)

  jitVsGlow(batchnorm_basic, inputs, running_mean, running_var)

# Test of the PyTorch batchnorm Node with weights and biases on Glow.
def test_batchnorm_with_weights():

  def batchnorm_with_weights(inputs, weight, bias, running_mean, running_var):
    return F.batch_norm(inputs, running_mean, running_var, weight=weight, bias=bias)

  inputs = torch.randn(1, 4, 5, 5)
  weight = torch.rand(4)
  bias = torch.rand(4)
  running_mean = torch.rand(4)
  running_var = torch.rand(4)
  
  jitVsGlow(batchnorm_with_weights, inputs, weight, bias, running_mean, running_var)