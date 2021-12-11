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


class SimpleNormModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(SimpleNormModule, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, tensor):
        return torch.norm(tensor, *self.args, **self.kwargs)


class TestNorm(utils.TorchGlowTestCase):
    def test_norm_basic(self):
        """Basic test of the PyTorch norm Node on Glow."""

        utils.compare_tracing_methods(
            SimpleNormModule(dim=0, p=2),
            torch.arange(8, dtype=torch.float).reshape(2, 4),
            fusible_ops={"aten::norm"},
        )

    def test_norm_float_p(self):
        """Test of the PyTorch norm Node that has p=2.0 on Glow."""

        utils.compare_tracing_methods(
            SimpleNormModule(dim=0, p=2.0),
            torch.arange(8, dtype=torch.float).reshape(2, 4),
            fusible_ops={"aten::norm"},
        )

    def test_norm_3d_inner_axis(self):
        """Basic test of the PyTorch norm Node on Glow."""

        utils.compare_tracing_methods(
            SimpleNormModule(dim=1),
            torch.arange(8, dtype=torch.float).reshape(2, 2, 2),
            fusible_ops={"aten::frobenius_norm"},
        )

    def test_norm_4d_outer_axis(self):
        """Basic test of the PyTorch norm Node on Glow."""

        utils.compare_tracing_methods(
            SimpleNormModule(dim=[3]),
            torch.arange(16, dtype=torch.float).reshape(2, 2, 2, 2),
            fusible_ops={"aten::frobenius_norm"},
        )

    def test_norm_keepdim(self):
        """Basic test of the PyTorch norm Node on Glow."""

        utils.compare_tracing_methods(
            SimpleNormModule(dim=[1], keepdim=True),
            torch.arange(16, dtype=torch.float).reshape(2, 4, 2),
            fusible_ops={"aten::frobenius_norm"},
        )
