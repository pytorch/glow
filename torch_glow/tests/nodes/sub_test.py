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


class SimpleSubtractModel(torch.nn.Module):
    def __init__(self):
        super(SimpleSubtractModel, self).__init__()

    def forward(self, a, b):
        if b.size() == torch.Size([]):
            return (a * a).sub(b.item())
        else:
            c = a.sub(b)
            return c.sub(c)


class TestSub(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", SimpleSubtractModel(), torch.randn(4), torch.randn(4)),
            lambda: (
                "broadcast_1",
                SimpleSubtractModel(),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            lambda: (
                "broadcast_2",
                SimpleSubtractModel(),
                torch.randn(8, 3, 4, 2),
                torch.randn(1, 2),
            ),
            lambda: (
                "broadcast_3",
                SimpleSubtractModel(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2),
            ),
            lambda: ("float", SimpleSubtractModel(), torch.randn(4), torch.tensor(3.9)),
            lambda: (
                "int",
                SimpleSubtractModel(),
                torch.randn(4),
                torch.tensor(20),
            ),
            lambda: (
                "int64",
                SimpleSubtractModel(),
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int64),
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int64),
            ),
        ]
    )
    def test_subtract(self, _, module, tensor, other):
        utils.run_comparison_tests(module, (tensor, other), fusible_ops={"aten::sub"})
