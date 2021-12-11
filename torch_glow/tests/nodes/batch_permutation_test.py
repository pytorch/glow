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


class SimpleBatchPermutationModule(torch.nn.Module):
    def forward(self, input, indices):
        return torch.ops._caffe2.BatchPermutation(input + input, indices)


class TestBatchPermutation(utils.TorchGlowTestCase):
    def test_batch_permutation_basic(self):
        """Basic test of the _caffe2::BatchPermutation Node on Glow."""

        x = torch.randn(4, 2, 3)
        indices = torch.tensor([1, 3, 0, 2], dtype=torch.int32)

        utils.compare_tracing_methods(
            SimpleBatchPermutationModule(),
            x,
            indices,
            fusible_ops={"_caffe2::BatchPermutation"},
        )
