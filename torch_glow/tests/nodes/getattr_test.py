# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch_glow
from tests.utils import GLOW_FUSION_GROUP, SUBGRAPH_ATTR


class TestGetAttr(unittest.TestCase):
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

            torch_glow.enableFusionPass()

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
