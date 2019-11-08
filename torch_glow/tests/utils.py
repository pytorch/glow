from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow

GLOW_NODE_NAME = "glow::FusionGroup"
SUBGRAPH_ATTR = "Subgraph"


def jitVsGlow(f, *inputs, expected_fused_ops, accept_all_ops=False):
    """
    Runs the given inputs *inputs on f both with and without lowering f to Glow,
    compares the results, and checks that ops in expected_fused_ops were indeed
    lowered to Glow.
    """
    jitVsGlow_(f, f, *inputs, expected_fused_ops=expected_fused_ops,
               accept_all_ops=accept_all_ops)


def jitVsGlow_(f_torch, f_glow, *inputs, expected_fused_ops=None,
               accept_all_ops=False):

    with torch.no_grad():
        torch_glow.disableFusionPass()
        torch_trace = torch.jit.trace(f_torch, inputs)
        torch_res = torch_trace(*inputs)

        torch_glow.enableFusionPass()
        glow_trace = torch.jit.trace(f_glow, inputs)
        glow_res = glow_trace(*inputs)

        # check that there are no Glow nodes in the torch graph
        torch_graph = torch_trace.graph_for(*inputs)
        print("torch_graph,", torch_graph)

        num_glow_nodes = len(torch_graph.findAllNodes(GLOW_NODE_NAME))
        assert num_glow_nodes == 0, "Expected no Glow nodes, found {}".format(
            num_glow_nodes
        )

        glow_graph = glow_trace.graph_for(*inputs)
        print("glow_graph,", glow_graph)

        expected_fused_ops_seen = set()

        # Whether or not at least one node was fused to Glow.
        nodes_were_fused = False

        # Check that ops that were *not* fused are *not* in expected_fused_ops
        for node in glow_graph.nodes():
            kind = node.kind()
            if kind != GLOW_NODE_NAME:
                # If the node is not a Glow fusion group, check that it is
                # *not* in expected_fused_ops
                assert accept_all_ops or kind not in expected_fused_ops, \
                    "Expected {} to be fused".format(kind)
            else:
                # If the node is a Glow fusion group, record which ops from
                # expected_fused_ops were in it

                # Get the definition of the fusion group
                glow_group = node.g(SUBGRAPH_ATTR)

                # Put all nodes that are in the group and in expected_fused_ops
                # into expected_fused_ops_seen
                for fused_node in glow_group.nodes():
                    nodes_were_fused = True
                    fused_node_kind = fused_node.kind()

                    if accept_all_ops or fused_node_kind in expected_fused_ops:
                        expected_fused_ops_seen.add(fused_node_kind)

        assert nodes_were_fused, "Expected some nodes to be fused to Glow"

        # If the sizes of expected_fused_ops and expected_fused_ops_seen are
        # different, some ops in expected_fused_ops are not in the graph at all
        assert accept_all_ops or len(expected_fused_ops) == len(expected_fused_ops_seen), \
            "Expected all of expected_fused_ops to be in the graph"

        if isinstance(torch_res, tuple) or isinstance(glow_res, tuple):
            assert isinstance(torch_res, tuple) and isinstance(glow_res, tuple)
            assert len(torch_res) == len(glow_res)
            for i in range(len(torch_res)):
                assert torch.allclose(torch_res[i], glow_res[i], atol=01e-6)
        else:
            is_all_close = torch.allclose(torch_res, glow_res, atol=01e-6)
            if not is_all_close:
                print("torch_res\n", torch_res)
                print("glow_res\n", glow_res)
            assert is_all_close


def graph_contains_str(graph, substr):
    return graph.str().find(substr) >= 0
