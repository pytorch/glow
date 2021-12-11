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
from torch.ao.quantization import (
    DeQuantStub,
    QConfig,
    QuantStub,
    convert,
    fuse_modules,
    observer,
    prepare,
)


my_qconfig = QConfig(
    activation=observer.default_observer,
    weight=observer.HistogramObserver.with_args(dtype=torch.qint8, reduce_range=False),
)


class TestQuantizedBatchNorm3DRelu(utils.TorchGlowTestCase):
    def test_batchnorm_relu_basic(self):
        """
        Basic test of the PyTorch 3D batchnorm RELU Node on Glow.
        """

        class SimpleQuantizedBatchNormRelu(nn.Module):
            def __init__(self, w, b, m, v):
                super(SimpleQuantizedBatchNormRelu, self).__init__()
                self.bn = torch.nn.BatchNorm3d(4)
                self.relu = torch.nn.ReLU()
                self.bn.weight = torch.nn.Parameter(w)
                self.bn.bias = torch.nn.Parameter(b)
                self.bn.running_mean = m
                self.bn.running_var = v
                self.q = QuantStub()
                self.dq = DeQuantStub()

            def forward(self, x):
                qx = self.q(x)
                qy = self.bn(qx)
                qy_relu = self.relu(qy)
                y = self.dq(qy_relu)
                return y

        C = 4
        weight = torch.ones(C) + torch.rand(C) * 0.001
        bias = torch.rand(C) * 0.0001
        running_mean = torch.zeros(C)
        running_var = torch.ones(C)

        inputs = torch.randn((10, C, 2, 3, 4), requires_grad=False)
        model = SimpleQuantizedBatchNormRelu(weight, bias, running_mean, running_var)
        model.eval()
        model.qconfig = my_qconfig
        modules_to_fuse = [["bn", "relu"]]
        fuse_modules(model, modules_to_fuse, inplace=True)
        prepare(model, inplace=True)
        model.forward(inputs)
        convert(model, inplace=True)

        # Because of the difference of quantization between PyTorch & Glow
        # We set eps big enough.
        # Batchnorm introduced great accuracy issues, which could create up to
        # ~1e-2 difference in some rare cases. In order to prevent this test
        # to be flaky, atol is set to be 0.1 and rtol is set to 0.00001.
        utils.compare_tracing_methods(
            model,
            inputs,
            fusible_ops={"quantized::batch_norm3d_relu"},
            atol=1e-1,
            rtol=1e-5,
            skip_to_glow=True,
        )
