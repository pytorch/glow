from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from tests import utils
from torch.quantization import QConfig, observer


my_qconfig = QConfig(
    activation=observer.default_observer,
    weight=observer.HistogramObserver.with_args(dtype=torch.qint8, reduce_range=False),
)


class TestQuantizedBatchNorm3D(utils.TorchGlowTestCase):
    def test_batchnorm_basic(self):
        """
        Basic test of the PyTorch 3D batchnorm Node on Glow.
        """

        class SimpleQuantizedBatchNorm(nn.Module):
            def __init__(
                self,
                C,
                running_mean,
                running_var,
                in_scale,
                in_zero_point,
                out_scale,
                out_zero_point,
            ):
                super(SimpleQuantizedBatchNorm, self).__init__()
                self.qconfig = my_qconfig
                self.batchnorm = nn.BatchNorm3d(C)
                self.batchnorm.scale = out_scale
                self.batchnorm.zero_point = out_zero_point
                self.batchnorm.running_mean = running_mean
                self.batchnorm.running_var = running_var
                self.relu = torch.nn.ReLU()
                self.q = torch.quantization.QuantStub()
                self.q.scale = in_scale
                self.q.zero_point = in_zero_point
                self.dq = torch.quantization.DeQuantStub()

            def forward(self, x):
                qx = self.q(x)
                qy = self.batchnorm(qx)
                y = self.dq(qy)
                return y

        C = 4
        in_scale = 0.123
        out_scale = 0.004
        in_zero_point = 90
        out_zero_point = 4
        running_mean = torch.zeros(C)
        running_var = torch.ones(C)

        inputs = torch.randn((5, C, 6, 32, 73), requires_grad=False)
        model = SimpleQuantizedBatchNorm(
            C,
            running_mean,
            running_var,
            in_scale,
            in_zero_point,
            out_scale,
            out_zero_point,
        )
        model.eval()

        utils.compare_tracing_methods(
            model,
            inputs,
            skip_to_glow=True,
        )

    def test_batchnorm_with_weights(self):
        """
        Test of the PyTorch 2D batchnorm Node with weights and biases on Glow.
        """

        class SimpleQuantizedBatchNorm(nn.Module):
            def __init__(
                self,
                C,
                weight,
                bias,
                running_mean,
                running_var,
                in_scale,
                in_zero_point,
                out_scale,
                out_zero_point,
            ):
                super(SimpleQuantizedBatchNorm, self).__init__()
                self.qconfig = my_qconfig
                self.batchnorm = nn.BatchNorm3d(C)
                self.batchnorm.scale = out_scale
                self.batchnorm.zero_point = out_zero_point
                self.batchnorm.weight = nn.Parameter(weight)
                self.batchnorm.bias = nn.Parameter(bias)
                self.batchnorm.running_mean = running_mean
                self.batchnorm.running_var = running_var
                self.relu = nn.ReLU()
                self.q = torch.quantization.QuantStub()
                self.q.scale = in_scale
                self.q.zero_point = in_zero_point
                self.dq = torch.quantization.DeQuantStub()

            def forward(self, x):
                qx = self.q(x)
                qy = self.batchnorm(qx)
                y = self.dq(qy)
                return y

        C = 7
        in_scale = 0.0031
        out_scale = 0.0047
        in_zero_point = -42
        out_zero_point = 23
        weight = torch.ones(C) + torch.rand(C) * 0.001
        bias = torch.rand(C) * 0.0001
        running_mean = torch.zeros(C)
        running_var = torch.ones(C)

        inputs = torch.randn((6, C, 4, 33, 42), requires_grad=False)
        model = SimpleQuantizedBatchNorm(
            C,
            weight,
            bias,
            running_mean,
            running_var,
            in_scale,
            in_zero_point,
            out_scale,
            out_zero_point,
        )
        model.eval()

        utils.compare_tracing_methods(
            model,
            inputs,
            skip_to_glow=True,
        )
