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


class SimpleTypeasModel(torch.nn.Module):
    def __init__(self):
        super(SimpleTypeasModel, self).__init__()

    def forward(self, tensor, other=None):
        # TODO: Understand and document the utility of the self-conversion test
        # as well as the additional tensor + tensor step
        other = tensor if other is None else other
        if tensor.dtype != torch.bool:
            tensor = tensor + tensor
        typed = tensor.type_as(other)
        return typed + typed


class TestTypeAs(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "to_int32",
                SimpleTypeasModel(),
                torch.randn(4),
                torch.zeros(4, dtype=torch.int32),
            ),
            lambda: (
                "from_int32",
                SimpleTypeasModel(),
                torch.randn(4).to(dtype=torch.int32),
                torch.zeros(4),
            ),
            lambda: (
                "from_bool",
                SimpleTypeasModel(),
                torch.randn(4).to(dtype=torch.bool),
                torch.zeros(4),
            ),
            lambda: ("self", SimpleTypeasModel(), torch.randn(4), None, False),
            lambda: (
                "f2f2",
                SimpleTypeasModel(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2),
                False,
            ),
            lambda: (
                "f2i2",
                SimpleTypeasModel(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2).to(dtype=torch.int32),
            ),
        ]
    )
    def test_typeas(self, _, module, tensor, other=None, should_fuse=True):
        if other is not None:
            utils.compare_tracing_methods(
                module,
                tensor,
                other,
                fusible_ops={"aten::type_as"} if should_fuse else {},
            )
        else:
            utils.compare_tracing_methods(
                module, tensor, fusible_ops={"aten::type_as"} if should_fuse else {}
            )
