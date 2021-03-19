from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleDynQLinearModule(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleDynQLinearModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = torch.nn.quantized.dynamic.Linear(self.in_features, self.out_features)

    def forward(self, input):
        return self.m(input)


class SimpleDynQLinearPerChannelModule(torch.nn.Module):
    def __init__(self, in_features, out_features, qconfig):
        super(SimpleDynQLinearPerChannelModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mf = torch.nn.Linear(self.in_features, self.out_features)
        self.mf.qconfig = qconfig
        self.m = torch.nn.quantized.dynamic.Linear.from_float(self.mf)

    def forward(self, input):
        return self.m(input)


class TestLinear(utils.TorchGlowTestCase):
    def test_linear_basic(self):
        """Basic test of the PyTorch aten::linear op on Glow."""

        n = 5
        in_features = 4
        out_features = 3

        input = torch.randn(n, in_features, dtype=torch.float)

        utils.compare_tracing_methods(
            SimpleDynQLinearModule(in_features, out_features),
            input,
            fusible_ops={"quantized::linear_dynamic"},
            fp16=True,
            skip_to_glow=True,
            rtol=7e-2,
            atol=7e-2,
        )

    def test_linear_per_channel(self):
        """Basic test of the PyTorch channel wise aten::linear op on Glow."""

        n = 5
        in_features = 3
        out_features = 4

        input = torch.randn(n, in_features, dtype=torch.float)
        my_qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_dynamic_quant_observer,
            weight=torch.quantization.default_per_channel_weight_observer,
        )

        utils.compare_tracing_methods(
            SimpleDynQLinearPerChannelModule(in_features, out_features, my_qconfig),
            input,
            fusible_ops={"quantized::linear_dynamic"},
            fp16=True,
            skip_to_glow=True,
            rtol=7e-2,
            atol=7e-2,
        )
