# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import torch_glow
import torch

from tests.utils import GLOW_NODE_NAME, SUBGRAPH_ATTR
import unittest


class TestAllowList(unittest.TestCase):
    def test_op_blacklist_allowlist(self):
        """Test Glow fuser allowlist overwrites blacklist mechanism."""

        def f(a, b):
            return (a + b) * (a - b)

        torch_glow.enableFusionPass()
        torch_glow.setFusionBlacklist(["aten::add", "aten::sub"])
        torch_glow.setFusionOverrideAllowlist(["aten::sub"])

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
        torch_glow.clearFusionOverrideAllowlist()

    def test_op_index_blacklist_allowlist(self):
        """Test Glow fuser allowlist overwrites index blacklisting mechanism."""

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

        torch_glow.enableFusionPass()
        # Only one div is allowed by index
        torch_glow.setFusionStartIndex(5)
        torch_glow.setFusionEndIndex(6)
        # But all divs are allowed by allowlist
        torch_glow.setFusionOverrideAllowlist(["aten::div"])

        a = torch.randn(5, 5)
        b = torch.randn(5, 5)

        jit_f = torch.jit.trace(f, (a, b))

        jit_f_graph = jit_f.graph_for(a, b)

        torch_glow.clearFusionIndices()
        torch_glow.clearFusionOverrideAllowlist()

        fused_muls = 0
        fused_divs = 0
        for node in jit_f_graph.nodes():
            if node.kind() == GLOW_NODE_NAME:
                glow_subgraph = node.g(SUBGRAPH_ATTR)
                for node in glow_subgraph.nodes():
                    if node.kind() == "aten::mul":
                        fused_muls += 1
                    if node.kind() == "aten::div":
                        fused_divs += 1

        assert fused_muls == 0, "Expected no aten::muls to be fused"
        assert fused_divs == 3, "Expected all 3 aten::divs to be fused"
