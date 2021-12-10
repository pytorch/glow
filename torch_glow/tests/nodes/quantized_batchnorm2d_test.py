# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from tests import utils
from torch.ao.quantization import QConfig, observer


my_qconfig = QConfig(
    activation=observer.default_observer,
    weight=observer.HistogramObserver.with_args(dtype=torch.qint8, reduce_range=False),
)


class TestQuantizedBatchNorm2D(utils.TorchGlowTestCase):
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
                self.batchnorm = nn.BatchNorm2d(C)
                self.batchnorm.scale = out_scale
                self.batchnorm.zero_point = out_zero_point
                self.batchnorm.running_mean = running_mean
                self.batchnorm.running_var = running_var
                self.relu = torch.nn.ReLU()
                self.q = torch.ao.quantization.QuantStub()
                self.q.scale = in_scale
                self.q.zero_point = in_zero_point
                self.dq = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                qx = self.q(x)
                qy = self.batchnorm(qx)
                qy = self.relu(qy)
                y = self.dq(qy)
                return y

        C = 7
        in_scale = 0.102
        out_scale = 0.003
        in_zero_point = -37
        out_zero_point = 3
        running_mean = torch.zeros(C)
        running_var = torch.ones(C)

        inputs = torch.randn((6, C, 43, 52), requires_grad=False)
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
                self.batchnorm = nn.BatchNorm2d(C)
                self.batchnorm.scale = out_scale
                self.batchnorm.zero_point = out_zero_point
                self.batchnorm.weight = nn.Parameter(weight)
                self.batchnorm.bias = nn.Parameter(bias)
                self.batchnorm.running_mean = running_mean
                self.batchnorm.running_var = running_var
                self.relu = nn.ReLU()
                self.q = torch.ao.quantization.QuantStub()
                self.q.scale = in_scale
                self.q.zero_point = in_zero_point
                self.dq = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                qx = self.q(x)
                qy = self.batchnorm(qx)
                y = self.dq(qy)
                return y

        C = 11
        in_scale = 0.234
        out_scale = 0.003
        in_zero_point = -10
        out_zero_point = -5
        weight = torch.ones(C) + torch.rand(C) * 0.001
        bias = torch.rand(C) * 0.0001
        running_mean = torch.zeros(C)
        running_var = torch.ones(C)

        inputs = torch.randn((6, C, 33, 42), requires_grad=False)
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
