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
from tests import utils


class SimpleAddMmModule(torch.nn.Module):
    def __init__(self, alpha=1, beta=1):
        super(SimpleAddMmModule, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, a, b, c):
        return (a + a).addmm(b, c)


class TestAddMM(utils.TorchGlowTestCase):
    def test_addmm_basic(self):
        """Basic test of the PyTorch addmm Node on Glow."""
        utils.run_comparison_tests(
            SimpleAddMmModule(),
            (torch.randn(6, 4), torch.randn(6, 10), torch.randn(10, 4)),
            fusible_ops={"aten::add", "aten::mm"},
            fp16vfp16_atol=1e-3,
            fp16vfp16_rtol=1e-3,
        )

    def test_addmm_broadcast(self):
        """Test of the PyTorch addmm with broadcasting add on Glow."""
        utils.run_comparison_tests(
            SimpleAddMmModule(),
            (torch.randn(4), torch.randn(6, 10), torch.randn(10, 4)),
            fusible_ops={"aten::add", "aten::mm"},
        )

    def test_addmm_broadcast_with_alpha_and_beta(self):
        """Test of the PyTorch addmm with broadcasting add on Glow."""
        utils.run_comparison_tests(
            SimpleAddMmModule(2.0, 3.0),
            (torch.randn(4), torch.randn(6, 10), torch.randn(10, 4)),
            fusible_ops={"aten::add", "aten::mm"},
        )
