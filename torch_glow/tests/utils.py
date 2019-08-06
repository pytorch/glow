from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow

GLOW_NODE_NAME = "glow::FusionGroup"
SUBGRAPH_ATTR = "Subgraph"


def jitVsGlow(f, *inputs, expected_fused_ops=None):
    """
    Runs the given inputs *inputs on f both with and without lowering f to Glow,
    compares the results, and checks that ops in expected_fused_ops were indeed
    lowered to Glow.
    """
    jitVsGlow_(f, f, *inputs, expected_fused_ops=expected_fused_ops)


def jitVsGlow_(f_torch, f_glow, *inputs, expected_fused_ops):
    assert (
        expected_fused_ops is not None and len(expected_fused_ops) > 0
    ), "Must pass non-empty list of ops that are expected to be fused"

    with torch.no_grad():
        torch_glow.disableFusionPass()
        torch_trace = torch.jit.trace(f_torch, inputs)
        torch_res = torch_trace(*inputs)

        torch_glow.enableFusionPass()
        glow_trace = torch.jit.trace(f_glow, inputs)
        glow_res = glow_trace(*inputs)

        # check that there are no Glow nodes in the torch graph
        torch_graph = torch_trace.graph_for(*inputs)
        num_glow_nodes = len(torch_graph.findAllNodes(GLOW_NODE_NAME))
        assert num_glow_nodes == 0, "Expected no Glow nodes, found {}".format(
            num_glow_nodes
        )

        glow_graph = glow_trace.graph_for(*inputs)
        expected_fused_ops_seen = set()

        # Check that ops that were *not* fused are *not* in expected_fused_ops
        for node in glow_graph.nodes():
            kind = node.kind()
            if kind != GLOW_NODE_NAME:
                # If the node is not a Glow fusion group, check that it is
                # *not* in expected_fused_ops
                assert kind not in expected_fused_ops, \
                    "Expected {} to be fused".format(kind)
            else:
                # If the node is a Glow fusion group, record which ops from
                # expected_fused_ops were in it

                # Get the definition of the fusion group
                glow_group = node.g(SUBGRAPH_ATTR)

                # Put all nodes that are in the group and in expected_fused_ops
                # into expected_fused_ops_seen
                for fused_node in glow_group.nodes():
                    fused_node_kind = fused_node.kind()

                    if fused_node_kind in expected_fused_ops:
                        expected_fused_ops_seen.add(fused_node_kind)

        # If the sizes of expected_fused_ops and expected_fused_ops_seen are
        # different, some ops in expected_fused_ops are not in the graph at all
        assert len(expected_fused_ops) == len(expected_fused_ops_seen), \
            "Expected all of expected_fused_ops to be in the graph"
        assert torch.allclose(torch_res, glow_res, atol=01e-6)
