# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch_glow
import torch
from tests.utils import GLOW_NODE_NAME, SUBGRAPH_ATTR


class TestMaxFusionMergeSize(unittest.TestCase):
    def test_max_fusion_merge_size(self):
        """Test Glow fuser maximum fusion merge size mechanism."""

        def f(a):
            return a * a * a * a * a * a

        torch_glow.disableFusionPass()

        # Set maximum fusion merge size to 3 nodes so that the
        # graph will not fit into 1 node
        torch_glow.setMaxFusionMergeSize(3)

        a = torch.randn(5, 5)

        jit_f = torch.jit.trace(f, (a))
        jit_f_graph = jit_f.graph_for(a)

        # print("before: ", jit_f_graph)

        torch_glow.glowCustomFuseDebug_(jit_f_graph)

        # print("after: ", jit_f_graph)

        fusion_nodes = 0
        for node in jit_f_graph.nodes():
            if node.kind() == GLOW_NODE_NAME:
                fusion_nodes += 1

        assert fusion_nodes > 1, "Expected more than one fusion group to be created"

        torch_glow.setMaxFusionMergeSize(0)

    def test_max_fusion_merge_size_zero(self):
        """Test Glow fuser maximum fusion merge size mechanism set to zero."""

        def f(a):
            return a * a * a * a * a * a

        torch_glow.disableFusionPass()

        # Set maximum fusion merge size to 0 so that there is
        # no limit to fusion
        torch_glow.setMaxFusionMergeSize(0)

        a = torch.randn(5, 5)

        jit_f = torch.jit.trace(f, (a))
        jit_f_graph = jit_f.graph_for(a)

        # print("before: ", jit_f_graph)

        torch_glow.glowCustomFuseDebug_(jit_f_graph)

        # print("after: ", jit_f_graph)

        fusion_nodes = 0
        for node in jit_f_graph.nodes():
            if node.kind() == GLOW_NODE_NAME:
                fusion_nodes += 1

        assert fusion_nodes == 1, "Expected just one fusion group to be created"

        torch_glow.setMaxFusionMergeSize(0)
