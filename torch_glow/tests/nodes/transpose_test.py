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


class SimpleTransposeModel(torch.nn.Module):
    def __init__(self, dim0=None, dim1=None, inplace=False):
        super(SimpleTransposeModel, self).__init__()
        self.dims = (dim0, dim1) if dim0 and dim1 else None
        self.inplace = inplace

    def forward(self, tensor):
        t = tensor + tensor
        if self.dims:
            return t.transpose_(*self.dims) if self.inplace else t.transpose(*self.dims)
        else:
            return t.t_() if self.inplace else t.t()


class TestTranspose(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("2d", SimpleTransposeModel(), torch.randn(7, 4)),
            lambda: ("1d", SimpleTransposeModel(), torch.randn(7)),
            lambda: ("inplace", SimpleTransposeModel(inplace=True), torch.randn(7, 4)),
        ]
    )
    def test_t(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::t"})

    @utils.deterministic_expand(
        [
            lambda: ("simple", SimpleTransposeModel(1, 2), torch.randn(2, 3, 4)),
            lambda: (
                "inplace",
                SimpleTransposeModel(1, 2, inplace=True),
                torch.randn(2, 3, 4),
            ),
            lambda: ("neg_dim", SimpleTransposeModel(-2, -1), torch.randn(2, 3, 4)),
        ]
    )
    def test_transpose(self, _, module, tensor, reference=None):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::transpose"})

    @utils.deterministic_expand(
        [lambda: ("oob_neg_dim", SimpleTransposeModel(-2, -4), torch.randn(2, 3, 4))]
    )
    def test_transpose_failure(self, _, module, tensor):
        with self.assertRaises(IndexError):
            utils.compare_tracing_methods(
                module, tensor, fusible_ops={"aten::transpose"}
            )
