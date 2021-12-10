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

from typing import Optional

import torch
from tests import utils


class SimpleDivModule(torch.nn.Module):
    def __init__(self, rounding_mode: Optional[str] = None):
        super(SimpleDivModule, self).__init__()
        self.rounding_mode = rounding_mode

    def forward(self, a, b):
        rounding_mode = self.rounding_mode
        if True:  # until 3rd agr is implemented, then: rounding_mode is None:
            if b.size() == torch.Size([]):
                return (a * a).div(b.item())
            else:
                c = a.div(b)
                return c.div(c)
        else:
            if b.size() == torch.Size([]):
                return (a * a).div(b.item(), rounding_mode=rounding_mode)
            else:
                c = a.div(b, rounding_mode=rounding_mode)
                return c.div(c, rounding_mode=rounding_mode)


class TestDiv(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", SimpleDivModule(), torch.randn(4), torch.randn(4)),
            lambda: (
                "basic_rm_true",
                SimpleDivModule(rounding_mode="true"),
                torch.randn(4),
                torch.randn(4),
            ),
            lambda: (
                "basic_rm_trunc",
                SimpleDivModule(rounding_mode="trunc"),
                torch.randn(4),
                torch.randn(4),
            ),
            lambda: (
                "basic_rm_floor",
                SimpleDivModule(rounding_mode="floor"),
                torch.randn(4),
                torch.randn(4),
            ),
            lambda: (
                "broadcast",
                SimpleDivModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            lambda: (
                "broadcast_rm_true",
                SimpleDivModule(rounding_mode="true"),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            lambda: (
                "broadcast_rm_trunc",
                SimpleDivModule(rounding_mode="trunc"),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            lambda: (
                "broadcast_rm_floor",
                SimpleDivModule(rounding_mode="floor"),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            lambda: (
                "broadcast",
                SimpleDivModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(1, 2),
            ),
            lambda: (
                "broadcast",
                SimpleDivModule(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2),
            ),
            lambda: (
                "float_tensor",
                SimpleDivModule(),
                torch.randn(4),
                torch.tensor(3.9),
            ),
            lambda: (
                "int_tensor",
                SimpleDivModule(),
                torch.tensor([4]),
                torch.tensor([10]),
                {"NNPI"},  # skip_for_backends
            ),
            # This one will go through (a * a) / b.item() and b.item() is an integer.
            lambda: (
                "int_number",
                SimpleDivModule(),
                torch.tensor([4]),
                torch.tensor(10),
                {"NNPI"},  # skip_for_backends
            ),
            lambda: (
                "int64",
                SimpleDivModule(),
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int64),
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int64),
                {"NNPI"},  # skip_for_backends
            ),
        ]
    )
    def test_div(self, _, module, a, b, skip_for_backends={}):
        utils.run_comparison_tests(
            module,
            (a, b),
            fusible_ops={"aten::div"},
            skip_for_backends=skip_for_backends,
        )
