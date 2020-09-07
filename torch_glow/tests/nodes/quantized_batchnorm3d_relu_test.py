from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn as nn
from tests.utils import jitVsGlow
from torch.quantization import (
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


class TestQuantizedBatchNorm3DRelu(unittest.TestCase):
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
        # to be flaky, atol is set to be 0.1.
        jitVsGlow(
            model,
            inputs,
            expected_fused_ops={"quantized::batch_norm3d_relu"},
            atol=1e-1,
            use_fp16=True,
        )
