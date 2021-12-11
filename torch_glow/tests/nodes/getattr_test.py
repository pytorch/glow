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

# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from tests import utils
from tests.utils import GLOW_FUSION_GROUP, SUBGRAPH_ATTR


class TestGetAttr(utils.TorchGlowTestCase):
    def test_getattr(self):
        """Test fusion of the PyTorch prim::GetAttr Node into the Glow subgraph."""
        with torch.no_grad():

            class Model(torch.nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.linear = torch.nn.Linear(2, 1)

                def forward(self, x):
                    return self.linear(x)

            x = torch.tensor([2.0, 3.0])

            torch_glow.enableFusionPass_DO_NOT_USE_THIS()

            m = Model()
            jit_m = torch.jit.trace(m, x)
            jit_m_graph = jit_m.graph_for(x)

            # Ensure all prim::GetAttrs were fused and none were left out
            found_getattrs = False
            for node in jit_m_graph.nodes():
                kind = node.kind()
                assert (
                    kind != "prim::GetAttr"
                ), "Expected all prim::GetAttrsGlow to be in Glow subgraph"
                if kind == GLOW_FUSION_GROUP:
                    glow_subgraph = node.g(SUBGRAPH_ATTR)
                    for node in glow_subgraph.nodes():
                        if node.kind() == "prim::GetAttr":
                            found_getattrs = True

            assert (
                found_getattrs
            ), "Expected to find prim::GetAttrs in the Glow subgraph"
