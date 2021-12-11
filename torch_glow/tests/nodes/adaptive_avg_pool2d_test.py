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


class SimpleAdapativeAvgPool2dModule(torch.nn.Module):
    def __init__(self, output_size):
        super(SimpleAdapativeAvgPool2dModule, self).__init__()
        self.output_size = output_size

    def forward(self, inputs):
        return F.adaptive_avg_pool2d(inputs, self.output_size)


class TestAdaptiveAvgPool2d(utils.TorchGlowTestCase):
    def test_adaptive_avg_pool2d_basic(self):
        """Basic test of PyTorch adaptive_avg_pool2d Node."""
        inputs = torch.randn(3, 6, 14, 14)

        utils.run_comparison_tests(
            SimpleAdapativeAvgPool2dModule((5, 5)),
            inputs,
            fusible_ops={"aten::adaptive_avg_pool2d"},
        )

    def test_adaptive_avg_pool2d_nonsquare_inputs(self):
        """Test of PyTorch adaptive_avg_pool2d Node with non-square inputs."""

        inputs = torch.randn(3, 6, 13, 14)

        utils.run_comparison_tests(
            SimpleAdapativeAvgPool2dModule((3, 3)),
            inputs,
            fusible_ops={"aten::adaptive_avg_pool2d"},
        )

    def test_adaptive_avg_pool2d_nonsquare_outputs(self):
        """Test of PyTorch adaptive_avg_pool2d Node with non-square outputs."""

        inputs = torch.randn(3, 6, 14, 14)

        utils.run_comparison_tests(
            SimpleAdapativeAvgPool2dModule((5, 3)),
            inputs,
            fusible_ops={"aten::adaptive_avg_pool2d"},
        )
