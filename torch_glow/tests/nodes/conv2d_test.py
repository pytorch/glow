import torch
import torch.nn.functional as F
import torch_glow
from collections import namedtuple

from tests.utils import jitVsGlow

# Basic test of the PyTorch conv2d Node on Glow.
def test_conv2d_basic():

  def conv2d_basic(inputs, filters):
        conv = F.conv2d(inputs, filters, padding=1)
        return F.relu(conv)

  inputs = torch.randn(1, 4, 5, 5)
  filters = torch.randn(8, 4, 3, 3)

  jitVsGlow(conv2d_basic, inputs, filters)

# Test of the PyTorch conv2d Node with a provided bias tensor.
def test_conv2d_with_bias():

  def conv2d_with_bias(inputs, filters, bias):
        conv = F.conv2d(inputs, filters, bias)
        return F.relu(conv)

  inputs = torch.randn(1, 4, 5, 5)
  filters = torch.randn(8, 4, 3, 3)
  bias = torch.randn(8)

  jitVsGlow(conv2d_with_bias, inputs, filters, bias)

# Test of the PyTorch conv2d Node sweeping through various parameters of the
# Node to test that they work correctly.
def test_conv2d_param_sweep():
  hwOpts = [3, 4]
  padOpts = [0, 1]
  groupsOpts = [1, 2]
  dilationOpts = [1, 2]
  strideOpts = [1, 2]

  Setting = namedtuple('Setting', ['h', 'w', 'p', 'g', 'd', 's',])

  settings = [Setting(h=h, w=w, p=p, g=g, d=d, s=s) for h in hwOpts for w in hwOpts for p in padOpts for g in groupsOpts for d in dilationOpts for s in strideOpts]

  for setting in settings:
      def conv2d_param_sweep(inputs, filters):
            conv = F.conv2d(inputs, filters, padding=setting.p, groups=setting.g)
            return F.relu(conv)

      inputs = torch.randn(2, 4, setting.h, setting.w)
      filters = torch.randn(8, 4/setting.g, 3, 3)

      jitVsGlow(conv2d_param_sweep, inputs, filters)
