from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F

from tests.utils import jitVsGlow


def test_avg_pool2d_basic():
    """Basic test of the PyTorch avg_pool2d Node on Glow."""
    def test_f(inputs):
        return F.avg_pool2d(inputs, 3)

    inputs = torch.randn(1, 4, 5, 5)

    jitVsGlow(test_f, inputs)


def test_avg_pool2d_with_args():
    """Test of the PyTorch avg_pool2d Node with arguments on Glow."""
    def test_f(inputs):
        return F.avg_pool2d(inputs, padding=3, kernel_size=7)

    inputs = torch.randn(1, 4, 10, 10)

    jitVsGlow(test_f, inputs)
