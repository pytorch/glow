from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from tests import utils


class SimpleAttentionModule(torch.nn.Module):
    def __init__(self):
        super(SimpleAttentionModule, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(32, 8)

    def forward(self, inputs):
        return self.self_attn(inputs, inputs, inputs)


class TestAttention(utils.TorchGlowTestCase):
    def test_attention_basic(self):
        """Basic test of the PyTorch attention Node on Glow."""
        inputs = torch.randn(2, 4, 32)
        model = SimpleAttentionModule()
        model.eval()
        torch_glow.enable_ignore_div_rounding_args()

        utils.compare_tracing_methods(
            model,
            inputs,
            fusible_ops={
                "aten::div",
                "aten::mul",
                "aten::transpose",
                "aten::softmax",
            },
        )
