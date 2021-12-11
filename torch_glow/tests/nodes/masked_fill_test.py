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


class SimpleMaskedFillModule(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleMaskedFillModule, self).__init__()
        self.inplace = inplace

    def forward(self, tensor, mask):
        if self.inplace:
            other = tensor + tensor
            other.masked_fill_(mask, 42.0)
            return other
        else:
            return torch.masked_fill(tensor + tensor, mask, 42.0)


class TestMaskedFill(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "basic",
                SimpleMaskedFillModule(),
                torch.randn([3]),
                torch.tensor([True, False, True], dtype=torch.bool),
            ),
            lambda: (
                "broadcasted_unit_dim",
                SimpleMaskedFillModule(),
                torch.randn([4, 1, 3]),
                torch.tensor([True, True, True], dtype=torch.bool),
            ),
            lambda: (
                "broadcasted_multi_dim",
                SimpleMaskedFillModule(),
                torch.randn([2, 4, 3, 3]),
                torch.tensor(
                    [[[[True, False, True]]], [[[True, False, True]]]], dtype=torch.bool
                ),
            ),
            lambda: (
                "inplace",
                SimpleMaskedFillModule(True),
                torch.randn([3]),
                torch.tensor([True, False, True], dtype=torch.bool),
            ),
        ]
    )
    def test_masked_fill(self, _, module, tensor, mask):
        utils.compare_tracing_methods(
            module, tensor, mask, fusible_ops={"aten::masked_fill"}
        )
