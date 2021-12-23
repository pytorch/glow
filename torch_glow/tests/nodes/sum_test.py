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


class SimpleSumModule(torch.nn.Module):
    def __init__(self, dtype=None):
        super(SimpleSumModule, self).__init__()
        self.dtype = dtype

    def forward(self, a):
        b = a + a
        return torch.sum(b, dtype=self.dtype)


class KeepdimSumModule(torch.nn.Module):
    def __init__(self, axis, keepdim, dtype=None):
        super(KeepdimSumModule, self).__init__()
        self.axis = axis
        self.keepdim = keepdim
        self.dtype = dtype

    def forward(self, a):
        b = a + a
        return torch.sum(b, self.axis, keepdim=self.keepdim, dtype=self.dtype)


class TestSumBasic(utils.TorchGlowTestCase):
    def test_sum_basic(self):
        a = torch.randn(2, 3, 4)

        utils.compare_tracing_methods(SimpleSumModule(), a, fusible_ops={"aten::sum"})


class TestSumKeepdim(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("keepdim", KeepdimSumModule(0, True), torch.randn(2, 3, 4)),
            lambda: ("axis_1", KeepdimSumModule(1, False), torch.randn(4, 3, 4)),
            lambda: (
                "axis_2_keepdim_f16",
                KeepdimSumModule(2, True, torch.float16),
                torch.randn(5, 2, 4),
            ),
            lambda: (
                "axis_1_f16",
                KeepdimSumModule(1, False, torch.float16),
                torch.randn(3, 1, 2),
            ),
            lambda: (
                "neg_axis_f16",
                KeepdimSumModule(-2, False, torch.float16),
                torch.randn(3, 1, 2),
            ),
            lambda: (
                "neg_axis_keepdim",
                KeepdimSumModule(-2, True),
                torch.randn(3, 1, 2),
            ),
            lambda: (
                "multiple_axes",
                KeepdimSumModule((0, 1), False, torch.float16),
                torch.randn(3, 4, 2),
            ),
            lambda: (
                "multiple_axes_keep_dim",
                KeepdimSumModule((2, 1), True, torch.float16),
                torch.randn(3, 4, 2),
            ),
        ]
    )
    def test_sum(self, _, module, a):
        utils.compare_tracing_methods(module, a, fusible_ops={"aten::sum"})
