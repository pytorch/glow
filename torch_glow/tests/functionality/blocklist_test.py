# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from tests import utils
from tests.utils import GLOW_FUSION_GROUP, SUBGRAPH_ATTR


class TestBlockList(utils.TorchGlowTestCase):
    def test_op_blocklist(self):
        """Test Glow fuser op kind blacklisting mechanism."""

        def f(a, b):
            return (a + b) * (a - b)

        torch_glow.enableFusionPass_DO_NOT_USE_THIS()
        torch_glow.setFusionBlocklist(["aten::add"])

        a = torch.randn(5, 5)
        b = torch.randn(5, 5)

        jit_f = torch.jit.trace(f, (a, b))

        jit_f_graph = jit_f.graph_for(a, b)

        fused_add = False
        fused_sub = False
        for node in jit_f_graph.nodes():
            if node.kind() == GLOW_FUSION_GROUP:
                glow_subgraph = node.g(SUBGRAPH_ATTR)
                for node in glow_subgraph.nodes():
                    if node.kind() == "aten::add":
                        fused_add = True
                    if node.kind() == "aten::sub":
                        fused_sub = True

        assert not fused_add, "Expected aten::add to be blacklisted"
        assert fused_sub, "Expected aten::sub to not be blacklisted"

        torch_glow.clearFusionBlocklist()

    def test_op_index_blocklist(self):
        """Test Glow fuser index blacklisting mechanism."""

        def f(a, b):
            x1 = a * b
            x2 = x1 * b
            x3 = x2 * a
            x4 = x3 / b
            x5 = x4 / a
            x6 = x5 / b
            x7 = x6 * a
            x8 = x7 * b
            return x8

        torch_glow.enableFusionPass_DO_NOT_USE_THIS()
        torch_glow.setFusionStartIndex(3)
        torch_glow.setFusionEndIndex(6)

        a = torch.randn(5, 5)
        b = torch.randn(5, 5)

        jit_f = torch.jit.trace(f, (a, b))

        jit_f_graph = jit_f.graph_for(a, b)

        torch_glow.clearFusionIndices()

        fused_muls = 0
        fused_divs = 0
        for node in jit_f_graph.nodes():
            if node.kind() == GLOW_FUSION_GROUP:
                glow_subgraph = node.g(SUBGRAPH_ATTR)
                for node in glow_subgraph.nodes():
                    if node.kind() == "aten::mul":
                        fused_muls += 1
                    if node.kind() == "aten::div":
                        fused_divs += 1

        assert fused_muls == 0, "Expected no aten::muls to be fused"
        assert fused_divs == 3, "Expected all 3 aten::divs to be fused"
