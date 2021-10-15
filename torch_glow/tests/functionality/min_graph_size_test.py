# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from tests import utils
from tests.utils import GLOW_FUSION_GROUP


class TestMinGraphSize(utils.TorchGlowTestCase):
    def test_min_graph_size(self):
        """Test Glow fuser minimum fusion group size mechanism."""

        def f(a, b, c):
            return (a * a * a * a) / (b * b * b) / (c * c * c * c * c)

        torch_glow.disableFusionPass()

        # Disable aten::div so that each group of aten::mul nodes will be forced
        # into separate subgraphs
        torch_glow.setFusionBlocklist(["aten::div"])

        # Set minimum fusion group size to 3 nodes so that the smallest group which
        # contains only 2 nodes will not be created
        torch_glow.setMinFusionGroupSize(3)

        a = torch.randn(5, 5)
        b = torch.randn(5, 5)
        c = torch.randn(5, 5)

        jit_f = torch.jit.trace(f, (a, b, c))
        jit_f_graph = jit_f.graph_for(a, b, c)

        # print("before: ", jit_f_graph)

        torch_glow.glowCustomFuseDebug_(jit_f_graph)

        # print("after: ", jit_f_graph)

        fusion_nodes = 0
        for node in jit_f_graph.nodes():
            if node.kind() == GLOW_FUSION_GROUP:
                fusion_nodes += 1

        assert fusion_nodes == 2, "Expected smallest fusion group to not be created"

        torch_glow.clearFusionBlocklist()
        torch_glow.setMinFusionGroupSize(0)
