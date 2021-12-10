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


class SimpleReciprocalModel(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleReciprocalModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        other = tensor + tensor
        return other.reciprocal_() if self.inplace else torch.reciprocal(other)


class TestReciprocal(utils.TorchGlowTestCase):
    def test_reciprocal(self):
        """Test of the PyTorch reciprocal Node on Glow."""

        utils.compare_tracing_methods(
            SimpleReciprocalModel(), torch.randn(4), fusible_ops={"aten::reciprocal"}
        )

    def test_inplace_reciprocal(self):
        """Test of the PyTorch inplace reciprocal Node on Glow."""

        # Expect fuser to out-of-place the operator
        utils.compare_tracing_methods(
            SimpleReciprocalModel(inplace=True),
            torch.randn(4),
            fusible_ops={"aten::reciprocal"},
        )
