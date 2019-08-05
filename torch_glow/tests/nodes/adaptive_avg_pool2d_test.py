from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F

from tests.utils import jitVsGlow


def test_adaptive_avg_pool2d_basic():
    """Basic test of PyTorch adaptive_avg_pool2d Node."""

    def test_f(inputs):
        return F.adaptive_avg_pool2d(inputs, (5, 5))

    inputs = torch.randn(3, 6, 14, 14)

    jitVsGlow(test_f, inputs)


def test_adaptive_avg_pool2d_nonsquare_inputs():
    """Test of PyTorch adaptive_avg_pool2d Node with non-square inputs."""

    def test_f(inputs):
        return F.adaptive_avg_pool2d(inputs, (3, 3))

    inputs = torch.randn(3, 6, 13, 14)

    jitVsGlow(test_f, inputs)


def test_adaptive_avg_pool2d_nonsquare_outputs():
    """Test of PyTorch adaptive_avg_pool2d Node with non-square outputs."""

    def test_f(inputs):
        return F.adaptive_avg_pool2d(inputs, (5, 3))

    inputs = torch.randn(3, 6, 14, 14)

    jitVsGlow(test_f, inputs)
