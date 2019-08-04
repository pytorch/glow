from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow

GLOW_NODE_NAME = "glow::FusionGroup"


def jitVsGlow(f, *inputs):
    """
    Runs the given inputs *inputs on f both with and without lowering f to Glow
    and compares the results.
    """
    jitVsGlow_(f, f, *inputs)


def jitVsGlow_(f_torch, f_glow, *inputs):
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

        # check that there is exactly 1 Glow node in the glow graph
        glow_graph = glow_trace.graph_for(*inputs)
        print("glow_graph,", glow_graph)
        num_glow_nodes = len(glow_graph.findAllNodes(GLOW_NODE_NAME))
        assert num_glow_nodes == 1, "Expected exactly 1 Glow node, found {}".format(
            num_glow_nodes)

        assert torch.allclose(torch_res, glow_res, atol=01e-6)
