import torch
import torch.nn.functional as F
import torch_glow

from tests.utils import jitVsGlow

# Basic test of the PyTorch max_pool2d Node on Glow.


def test_max_pool2d_basic():
    def max_pool2d_basic(inputs):
        return F.max_pool2d(inputs, 3)

    inputs = torch.randn(1, 4, 5, 5)

    jitVsGlow(max_pool2d_basic, inputs)


# Test of the PyTorch max_pool2d Node with arguments on Glow.


def test_max_pool2d_with_args():
    def max_pool2d_with_args(inputs):
        return F.max_pool2d(inputs, padding=3, kernel_size=7)

    inputs = torch.randn(1, 4, 10, 10)

    jitVsGlow(max_pool2d_with_args, inputs)
