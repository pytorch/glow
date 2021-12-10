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
import torch.nn.functional as F
from tests import utils


class SimpleLinearModule(torch.nn.Module):
    def __init__(self):
        super(SimpleLinearModule, self).__init__()

    def forward(self, input, weight, bias=None):
        return F.linear((input + input), weight, bias)


class TestLinear(utils.TorchGlowTestCase):
    def test_linear_basic(self):
        """Basic test of the PyTorch aten::linear op on Glow."""

        def test_f(input, weight, bias=None):
            return F.linear((input + input), weight, bias)

        n = 5
        in_features = 4
        out_features = 3

        input = torch.randn(n, in_features)
        weight = torch.randn(out_features, in_features)

        utils.compare_tracing_methods(
            SimpleLinearModule(), input, weight, fusible_ops={"aten::linear"}
        )

    def test_linear_bias(self):
        """Test of the PyTorch aten::linear op on Glow."""

        def test_f(input, weight, bias=None):
            return F.linear((input + input), weight, bias)

        n = 5
        in_features = 4
        out_features = 3

        input = torch.randn(n, in_features)
        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)

        utils.compare_tracing_methods(
            SimpleLinearModule(), input, weight, bias, fusible_ops={"aten::linear"}
        )

    def test_linear_broadcast(self):
        """Test of the PyTorch aten::linear op with broadcasting on Glow."""

        def test_f(input, weight, bias=None):
            return F.linear((input + input), weight, bias)

        n = 5
        in_features = 4
        out_features = 3

        input = torch.randn(n, 9, 7, in_features)
        weight = torch.randn(out_features, in_features)

        utils.compare_tracing_methods(
            SimpleLinearModule(), input, weight, fusible_ops={"aten::linear"}
        )
