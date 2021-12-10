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


class SimpleArgSortModule(torch.nn.Module):
    def __init__(self, descending=True):
        super(SimpleArgSortModule, self).__init__()
        self.descending = descending

    def forward(self, inputs):
        # Only last dim is currently supported
        return torch.argsort(inputs, dim=-1, descending=self.descending)


class TestArgSort(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "desc",
                SimpleArgSortModule(),
                torch.randn(4),
            ),
            lambda: (
                "asc",
                SimpleArgSortModule(descending=False),
                torch.randn(4),
            ),
            lambda: (
                "2d_desc",
                SimpleArgSortModule(),
                torch.randn(4, 3),
            ),
            lambda: (
                "3d_asc",
                SimpleArgSortModule(descending=False),
                torch.randn(6, 4, 5),
            ),
            lambda: (
                "4d_desc",
                SimpleArgSortModule(),
                torch.randn(4, 7, 7, 3),
            ),
        ]
    )
    def test_argsort(self, _, module, a):
        utils.compare_tracing_methods(module, a, fusible_ops={"aten::argsort"})
