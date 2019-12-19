from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F

from tests.utils import jitVsGlow
import unittest


class TestBatchNorm(unittest.TestCase):
    def test_batchnorm_basic(self):
        """Basic test of the PyTorch batchnorm Node on Glow."""

        def test_f(inputs, running_mean, running_var):
            return F.batch_norm(inputs, running_mean, running_var)

        inputs = torch.randn(1, 4, 5, 5)
        running_mean = torch.rand(4)
        running_var = torch.rand(4)

        jitVsGlow(
            test_f,
            inputs,
            running_mean,
            running_var,
            expected_fused_ops={"aten::batch_norm"},
        )

    def test_batchnorm_with_weights(self):
        """Test of the PyTorch batchnorm Node with weights and biases on Glow."""

        def test_f(inputs, weight, bias, running_mean, running_var):
            return F.batch_norm(
                inputs, running_mean, running_var, weight=weight, bias=bias
            )

        inputs = torch.randn(1, 4, 5, 5)
        weight = torch.rand(4)
        bias = torch.rand(4)
        running_mean = torch.rand(4)
        running_var = torch.rand(4)

        jitVsGlow(
            test_f,
            inputs,
            weight,
            bias,
            running_mean,
            running_var,
            expected_fused_ops={"aten::batch_norm"},
        )
