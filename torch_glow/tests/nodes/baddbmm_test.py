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

# pyre-ignore-all-errors

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from glow.glow.torch_glow.tests.tests import utils


class SimpleBAddBmmModule(torch.nn.Module):
    def __init__(self, alpha=1, beta=1):
        super(SimpleBAddBmmModule, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, a, b, c):
        return (a + a).baddbmm(b, c)


class TestBAddBmm(utils.TorchGlowTestCase):
    def test_baddbmm_basic(self):
        """Basic test of the PyTorch baddbmm Node on Glow."""
        utils.run_comparison_tests(
            SimpleBAddBmmModule(),
            (torch.randn(3, 6, 4), torch.randn(3, 6, 10), torch.randn(3, 10, 4)),
            fusible_ops={"aten::baddbmm"},
        )

    def test_baddbmm_broadcast(self):
        """Test of the PyTorch baddbmm with broadcasting add on Glow."""
        utils.run_comparison_tests(
            SimpleBAddBmmModule(),
            (torch.randn(1, 4), torch.randn(3, 6, 10), torch.randn(3, 10, 4)),
            fusible_ops={"aten::baddbmm"},
        )

    def test_baddbmm_broadcast_with_alpha_and_beta(self):
        """Test of the PyTorch baddbmm with broadcasting add on Glow, a=2/b=3"""
        utils.run_comparison_tests(
            SimpleBAddBmmModule(2.0, 3.0),
            (torch.randn(1, 4), torch.randn(3, 6, 10), torch.randn(3, 10, 4)),
            fusible_ops={"aten::baddbmm"},
        )

    def test_baddbmm_basic_tracing(self):
        """Basic test of the PyTorch baddbmm Node on Glow, w/ trace"""
        utils.compare_tracing_methods(
            SimpleBAddBmmModule(),
            torch.randn(2, 3, 5),
            torch.randn(2, 3, 9),
            torch.randn(2, 9, 5),
            fusible_ops={"aten::baddbmm"},
        )

    def test_baddbmm_broadcast_tracing(self):
        """Test of the PyTorch baddbmm with broadcasting add on Glow, w/ trace"""
        utils.compare_tracing_methods(
            SimpleBAddBmmModule(),
            torch.randn(1),
            torch.randn(3, 6, 9),
            torch.randn(3, 9, 5),
            fusible_ops={"aten::baddbmm"},
        )

    def test_baddbmm_broadcast_with_alpha_and_beta_tracing(self):
        """Test of the PyTorch baddbmm with broadcasting add on Glow, non-1 a/b, w/ trace"""
        utils.compare_tracing_methods(
            SimpleBAddBmmModule(0.5, 0.3),
            torch.randn(1),
            torch.randn(3, 6, 9),
            torch.randn(3, 9, 5),
            fusible_ops={"aten::baddbmm"},
        )

    def test_baddbmm_broadcast_tracing_error(self):
        """Test of the PyTorch baddbmm with broadcasting add on Glow, w/ trace + error"""
        utils.compare_tracing_methods_error(
            SimpleBAddBmmModule(),
            torch.randn(4),
            torch.randn(3, 6, 9),
            torch.randn(3, 9, 5),
            fusible_ops={"aten::baddbmm"},
        )
