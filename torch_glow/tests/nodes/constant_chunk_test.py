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


class SimpleChunkModel(torch.nn.Module):
    def __init__(self, chunks, dimension):
        super(SimpleChunkModel, self).__init__()
        self.chunks = chunks
        self.dimension = dimension

    def forward(self, input):
        return torch.chunk(input + input, self.chunks, self.dimension)


class TestConstantChunk(utils.TorchGlowTestCase):
    def test_constant_chunk_basic(self):
        """Test of prim::ConstantChunk node on glow"""

        x = torch.rand((10, 11))
        # shapes: [(10,4), (10,4), (10,3)]
        utils.compare_tracing_methods(
            SimpleChunkModel(3, 1),
            x,
            fusible_ops={"prim::ConstantChunk"},
            skip_to_glow=True,
        )

    def test_constant_chunk_negative_indices(self):
        """Test of prim::ConstantChunk node on glow"""

        x = torch.rand((10, 11))
        # shapes: [(4,11), (4,11), (2,11)]
        utils.compare_tracing_methods(
            SimpleChunkModel(3, -2),
            x,
            fusible_ops={"prim::ConstantChunk"},
            skip_to_glow=True,
        )
