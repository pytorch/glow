from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn as nn
from tests.utils import jitVsGlow
from torch.quantization import QConfig, observer


my_qconfig = QConfig(
    activation=observer.default_observer,
    weight=observer.HistogramObserver.with_args(dtype=torch.qint8, reduce_range=False),
)


class TestQuantizedBatchNorm3D(unittest.TestCase):
    def test_batchnorm_basic(self):
        """
        Basic test of the PyTorch 3D batchnorm Node on Glow.
        """

        class SimpleQuantizedBatchNorm(nn.Module):
            def __init__(self, C, running_mean, running_var, scale, zero_point):
                super(SimpleQuantizedBatchNorm, self).__init__()
                self.qconfig = my_qconfig
                self.batchnorm = nn.quantized.BatchNorm3d(C)
                self.batchnorm.scale = scale
                self.batchnorm.zero_point = zero_point
                self.batchnorm.running_mean = running_mean
                self.batchnorm.running_var = running_var
                self.relu = torch.nn.ReLU()
                self.dq = torch.nn.quantized.DeQuantize()

            def forward(self, x):
                return self.dq(self.relu(self.batchnorm(x)))

        C = 4
        in_scale = out_scale = 0.004
        in_zero_point = out_zero_point = 4
        running_mean = torch.zeros(C)
        running_var = torch.ones(C)

        inputs = torch.randn((5, C, 6, 32, 73), requires_grad=False)
        inputs = torch.quantize_per_tensor(
            inputs, scale=in_scale, zero_point=in_zero_point, dtype=torch.qint8
        )
        model = SimpleQuantizedBatchNorm(
            C, running_mean, running_var, out_scale, out_zero_point
        )
        model.eval()

        jitVsGlow(
            model, inputs, expected_fused_ops={"quantized::batch_norm3d"}, use_fp16=True
        )

    def test_batchnorm_with_weights(self):
        """
        Test of the PyTorch 2D batchnorm Node with weights and biases on Glow.
        """

        class SimpleQuantizedBatchNorm(nn.Module):
            def __init__(
                self, C, weight, bias, running_mean, running_var, scale, zero_point
            ):
                super(SimpleQuantizedBatchNorm, self).__init__()
                self.qconfig = my_qconfig
                self.batchnorm = nn.quantized.BatchNorm3d(C)
                self.batchnorm.scale = scale
                self.batchnorm.zero_point = zero_point
                self.batchnorm.weight = torch.nn.Parameter(weight)
                self.batchnorm.bias = torch.nn.Parameter(bias)
                self.batchnorm.running_mean = running_mean
                self.batchnorm.running_var = running_var
                self.relu = torch.nn.ReLU()
                self.dq = torch.nn.quantized.DeQuantize()

            def forward(self, x):
                return self.dq(self.relu(self.batchnorm(x)))

        C = 7
        in_scale = out_scale = 0.0047
        in_zero_point = out_zero_point = -7
        weight = torch.ones(C) + torch.rand(C) * 0.001
        bias = torch.rand(C) * 0.0001
        running_mean = torch.zeros(C)
        running_var = torch.ones(C)

        inputs = torch.randn(6, C, 4, 33, 42)
        inputs = torch.quantize_per_tensor(
            inputs, scale=in_scale, zero_point=in_zero_point, dtype=torch.qint8
        )
        model = SimpleQuantizedBatchNorm(
            C, weight, bias, running_mean, running_var, out_scale, out_zero_point
        )
        model.eval()

        jitVsGlow(
            model, inputs, expected_fused_ops={"quantized::batch_norm3d"}, use_fp16=True
        )
