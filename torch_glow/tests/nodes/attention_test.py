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
