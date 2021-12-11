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


class SimpleFlattenModule(torch.nn.Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super(SimpleFlattenModule, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return torch.flatten(input, start_dim=self.start_dim, end_dim=self.end_dim)


class TestFlatten(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", SimpleFlattenModule(), torch.randn(2, 3, 2, 5)),
            lambda: ("start_at_0", SimpleFlattenModule(0, 2), torch.randn(2, 3, 2, 5)),
            lambda: (
                "start_in_middle",
                SimpleFlattenModule(1, 2),
                torch.randn(2, 3, 2, 5),
            ),
            lambda: (
                "negative_end_dim",
                SimpleFlattenModule(0, -2),
                torch.randn(2, 3, 2, 5),
            ),
            lambda: ("same_dim", SimpleFlattenModule(2, 2), torch.randn(2, 3, 2, 5)),
            lambda: (
                "negative_start_dim",
                SimpleFlattenModule(-3, -1),
                torch.randn(2, 3, 2, 5),
            ),
        ]
    )
    def test_flatten(self, _, module, input):
        utils.compare_tracing_methods(module, input, fusible_ops={"aten::flatten"})
