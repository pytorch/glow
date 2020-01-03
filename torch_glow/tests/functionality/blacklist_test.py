from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import GLOW_NODE_NAME, SUBGRAPH_ATTR
import torch_glow
import unittest


class TestBlackList(unittest.TestCase):
    def test_op_blacklist(self):
        """Test Glow fuser blacklisting mechanism."""

        def f(a, b):
            return (a + b) * (a - b)

        torch_glow.enableFusionPass()
        torch_glow.setFusionBlacklist(["aten::add"])

        a = torch.randn(5, 5)
        b = torch.randn(5, 5)

        jit_f = torch.jit.trace(f, (a, b))

        jit_f_graph = jit_f.graph_for(a, b)

        fused_add = False
        fused_sub = False
        for node in jit_f_graph.nodes():
            if node.kind() == GLOW_NODE_NAME:
                glow_subgraph = node.g(SUBGRAPH_ATTR)
                for node in glow_subgraph.nodes():
                    if node.kind() == "aten::add":
                        fused_add = True
                    if node.kind() == "aten::sub":
                        fused_sub = True

        assert not fused_add, "Expected aten::add to be blacklisted"
        assert fused_sub, "Expected aten::sub to not be blacklisted"

        torch_glow.clearFusionBlacklist()
