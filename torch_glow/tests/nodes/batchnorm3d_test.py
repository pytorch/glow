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


class TestBatchNorm3D(utils.TorchGlowTestCase):
    def test_batchnorm_basic(self):
        """
        Basic test of the PyTorch 3D batchnorm Node on Glow.
        """

        class SimpleBatchNorm(nn.Module):
            def __init__(self, num_channels, running_mean, running_var):
                super(SimpleBatchNorm, self).__init__()
                self.batchnorm = nn.BatchNorm3d(num_channels)
                self.batchnorm.running_mean = running_mean
                self.batchnorm.running_var = running_var

            def forward(self, x):
                return self.batchnorm(x)

        num_channels = 4
        running_mean = torch.rand(num_channels)
        running_var = torch.rand(num_channels)
        model = SimpleBatchNorm(num_channels, running_mean, running_var)
        model.eval()

        inputs = torch.randn(1, num_channels, 4, 5, 5)
        utils.compare_tracing_methods(model, inputs, fusible_ops={"aten::batch_norm"})

    def test_batchnorm_with_weights(self):
        """
        Test of the PyTorch 3D batchnorm Node with weights and biases on Glow.
        """

        class SimpleBatchNorm(nn.Module):
            def __init__(self, num_channels, weight, bias, running_mean, running_var):
                super(SimpleBatchNorm, self).__init__()
                self.batchnorm = nn.BatchNorm3d(num_channels)
                self.batchnorm.weight = torch.nn.Parameter(weight)
                self.batchnorm.bias = torch.nn.Parameter(bias)
                self.batchnorm.running_mean = running_mean
                self.batchnorm.running_var = running_var

            def forward(self, x):
                return self.batchnorm(x)

        num_channels = 4
        weight = torch.rand(num_channels)
        bias = torch.rand(num_channels)
        running_mean = torch.rand(num_channels)
        running_var = torch.ones(num_channels)

        inputs = torch.randn(1, num_channels, 4, 5, 5)
        model = SimpleBatchNorm(num_channels, weight, bias, running_mean, running_var)
        model.eval()

        utils.compare_tracing_methods(model, inputs, fusible_ops={"aten::batch_norm"})
