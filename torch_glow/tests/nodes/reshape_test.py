from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow

from tests.utils import jitVsGlow


def jitScriptVsGlow(f_torch, f_glow, *inputs):

    with torch.no_grad():
        torch_glow.disableFusionPass()
        torch_script = f_torch
        if (not isinstance(f_torch, torch.jit.ScriptModule)):
            torch_srcipt = torch.jit.script(f_torch)
        torch_res = torch_script(*inputs)

        torch_glow.enableFusionPass()
        glow_script = f_glow
        if (not isinstance(f_glow, torch.jit.ScriptModule)):
            glow_script = torch.jit.script(f_glow)
        glow_res = glow_script(*inputs)

        torch_graph = torch_script.graph_for(*inputs)
        print("torch_graph, ", torch_graph)
        num_glow_nodes = len(torch_graph.findAllNodes(GLOW_NODE_NAME))
        assert num_glow_nodes == 0, "Expected no Glow nodes, found {}".format(num_glow_nodes)

        glow_graph = glow_script.graph_for(*inputs)
        print("glow_graph,", glow_graph)
        num_glow_nodes = len(glow_graph.findAllNodes(GLOW_NODE_NAME))
        assert num_glow_nodes == 1, "Expected exactly 1 Glow node, found {}".format(num_glow_nodes)

        assert torch.allclose(torch_res, glow_res, atol=01e-6)

def test_reshape_basic():

    @torch.jit.ScriptModule
    def reshape_basic(x, y, z):
        # type: (Tensor, Tensor, Tuple[int, int]) -> Tensor
        a = x.add(y)
        b = a.reshape(z)
        return b

    x = torch.randn(2, 10)
    y = torch.randn(2, 10)
    z = (4, 5)
    jitScriptVsGlow(reshape_basic, reshape_basic, x, y, z)
