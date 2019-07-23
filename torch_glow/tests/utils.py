import torch
import torch_glow

GLOW_NODE_NAME = "glow::CompilationGroup"

# Runs the given inputs \p *inputs on \p f both with and without lowering \p f
# to Glow and compares the results.
def jitVsGlow(f, *inputs):
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
    assert num_glow_nodes == 0, "Expected no Glow nodes, found {}".format(num_glow_nodes)

    # check that there is exactly 1 Glow node in the glow graph
    glow_graph = glow_trace.graph_for(*inputs)
    print("glow_graph,", glow_graph)
    num_glow_nodes = len(glow_graph.findAllNodes(GLOW_NODE_NAME))
    assert num_glow_nodes == 1, "Expected exactly 1 Glow node, found {}".format(num_glow_nodes)

    assert torch.allclose(torch_res, glow_res, atol=01e-6)

def jitScriptVsGlow(f_torch, f_glow, *inputs):
  with torch.no_grad():
    torch_glow.disableFusionPass()
    torch_script = f_torch
    if (not isinstance(f_torch, torch.jit.ScriptModule)):
      torch_script = torch.jit.script(f_torch)
    torch_res = torch_script(*inputs)

    torch_glow.enableFusionPass()
    glow_script = f_glow
    if (not isinstance(f_glow, torch.jit.ScriptModule)):
      glow_script = torch.jit.script(f_glow)
    glow_res = glow_script(*inputs)

    # check that there are no Glow nodes in the torch graph
    torch_graph = torch_script.graph_for(*inputs)
    print("torch_graph,", torch_graph)
    num_glow_nodes = len(torch_graph.findAllNodes(GLOW_NODE_NAME))
    assert num_glow_nodes == 0, "Expected no Glow nodes, found {}".format(num_glow_nodes)

    # check that there is exactly 1 Glow node in the glow graph
    glow_graph = glow_script.graph_for(*inputs)
    print("glow_graph,", glow_graph)
    num_glow_nodes = len(glow_graph.findAllNodes(GLOW_NODE_NAME))
    assert num_glow_nodes == 1, "Expected exactly 1 Glow node, found {}".format(num_glow_nodes)

    assert torch.allclose(torch_res, glow_res, atol=01e-6)
