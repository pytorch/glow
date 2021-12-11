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


class SimpleRsubModel(torch.nn.Module):
    def __init__(self):
        super(SimpleRsubModel, self).__init__()

    def forward(self, tensor, other):
        if other.size() == torch.Size([]):
            return torch.rsub((tensor * tensor), other.item())
        else:
            third = torch.rsub(tensor, other)
            return torch.rsub(third, third)


class TestRsub(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", SimpleRsubModel(), torch.randn(4), torch.randn(4)),
            lambda: (
                "broadcast",
                SimpleRsubModel(),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            lambda: (
                "broadcast",
                SimpleRsubModel(),
                torch.randn(8, 3, 4, 2),
                torch.randn(1, 2),
            ),
            lambda: (
                "broadcast",
                SimpleRsubModel(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2),
            ),
        ]
    )
    def test_rsub_as_sub(self, _, module, tensor, other):
        # aten::rsub is normalized as aten::sub
        utils.compare_tracing_methods(module, tensor, other, fusible_ops={"aten::sub"})

    @utils.deterministic_expand(
        [
            lambda: ("float", SimpleRsubModel(), torch.randn(4), torch.tensor(13.293)),
            lambda: ("int", SimpleRsubModel(), torch.randn(4), torch.tensor(4)),
        ]
    )
    def test_rsub(self, _, module, tensor, other):
        utils.compare_tracing_methods(module, tensor, other, fusible_ops={"aten::rsub"})
