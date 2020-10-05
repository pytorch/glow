# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import torch_glow
import torch

GLOW_NODE_NAME = "glow::FusionGroup"
SUBGRAPH_ATTR = "Subgraph"
BACKEND_NAME_KEY = "BACKEND_NAME"


def get_backend_name():
    if BACKEND_NAME_KEY in os.environ:
        return os.environ[BACKEND_NAME_KEY]
    else:
        return "Interpreter"


def check_skip(case):
    backend = get_backend_name()
    supported = {"Interpreter"}
    try:
        supported = supported | case.supported_backends
    except AttributeError:
        pass

    if backend not in supported:
        case.skipTest("Skipping tests for backend: " + backend)


def jitVsGlow(
    f,
    *inputs,
    expected_fused_ops,
    accept_all_ops=False,
    check_trace=True,
    atol=5e-4,
    rtol=1e-3,
    black_list=None,
    use_script=False,
    use_fp16=False,
    backend_name=None,
):
    """
    Runs the given inputs *inputs on f both with and without lowering f to Glow,
    compares the results, and checks that ops in expected_fused_ops were indeed
    lowered to Glow.
    """
    if use_script:
        scriptVsGlow(
            f,
            atol,
            rtol,
            *inputs,
            expected_fused_ops=expected_fused_ops,
            accept_all_ops=accept_all_ops,
            black_list=black_list,
            use_fp16=use_fp16,
            backend_name=backend_name,
        )
    else:
        traceVsGlow(
            f,
            f,
            check_trace,
            atol,
            rtol,
            *inputs,
            expected_fused_ops=expected_fused_ops,
            accept_all_ops=accept_all_ops,
            black_list=black_list,
            use_fp16=use_fp16,
            backend_name=backend_name,
        )


def checkResult(torch_res, glow_res, atol, rtol):
    if isinstance(torch_res, tuple) or isinstance(glow_res, tuple):
        assert isinstance(torch_res, tuple) and isinstance(glow_res, tuple)
        assert len(torch_res) == len(glow_res)
        for i in range(len(torch_res)):
            print("torch shape: {}".format(torch_res[i].shape), file=sys.stderr)
            print("glow shape: {}".format(glow_res[i].shape), file=sys.stderr)
            assert torch.allclose(torch_res[i], glow_res[i], atol=atol, rtol=rtol)
    else:
        print("torch shape: {}".format(torch_res.shape), file=sys.stderr)
        print("glow shape: {}".format(glow_res.shape), file=sys.stderr)
        is_all_close = torch.allclose(torch_res, glow_res, atol=atol, rtol=rtol)
        if not is_all_close:
            print("torch_res\n", torch_res)
            print("glow_res\n", glow_res)
            diff = torch.abs(glow_res - torch_res)
            print("diff\n", diff)
            print(
                "diff histogram (100 buckets from 0.0 to 1.0)\n",
                torch.histc(diff, bins=100, min=0, max=1),
            )
            print("max diff\n", torch.max(diff))
        assert is_all_close


def checkExpectedOps(glow_graph, expected_fused_ops, accept_all_ops):
    with torch.no_grad():
        expected_fused_ops_seen = set()

        # Whether or not at least one node was fused to Glow.
        nodes_were_fused = False

        # Check that ops that were *not* fused are *not* in expected_fused_ops
        for node in glow_graph.nodes():
            kind = node.kind()
            if kind != GLOW_NODE_NAME:
                # If the node is not a Glow fusion group, check that it is
                # *not* in expected_fused_ops
                assert (
                    accept_all_ops or kind not in expected_fused_ops
                ), "Expected {} to be fused".format(kind)
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
        assert accept_all_ops or len(expected_fused_ops) == len(
            expected_fused_ops_seen
        ), "Expected all of expected_fused_ops to be in the graph"


def traceVsGlow(
    f_torch,
    f_glow,
    check_trace,
    atol,
    rtol,
    *inputs,
    expected_fused_ops=None,
    accept_all_ops=False,
    black_list=None,
    use_fp16=False,
    backend_name=None,
):
    if black_list is None:
        black_list = []
    with torch.no_grad():
        torch_glow.disableFusionPass()

        torch_trace = torch.jit.trace(f_torch, inputs, check_trace=check_trace)
        torch_res = torch_trace(*inputs)

        torch_glow.enableFusionPass()
        torch_glow.setFusionBlacklist(black_list)

        if use_fp16:
            torch_glow.enable_convert_to_fp16()
            torch_glow.enable_convert_fused_to_fp16()
            torch_glow.enable_clip_fp16()
        else:
            torch_glow.disable_convert_to_fp16()
            torch_glow.disable_convert_fused_to_fp16()
            torch_glow.disable_clip_fp16()

        if backend_name:
            torch_glow.setGlowBackend(backend_name)
        else:
            torch_glow.setGlowBackend("Interpreter")

        glow_trace = torch.jit.trace(f_glow, inputs, check_trace=check_trace)
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

        # need to explicitly clear settings to avoid carry-over static settings
        torch_glow.disableFusionPass()
        torch_glow.disable_convert_to_fp16()
        torch_glow.disable_convert_fused_to_fp16()
        torch_glow.disable_clip_fp16()
        torch_glow.setGlowBackend("Interpreter")

    checkExpectedOps(glow_graph, expected_fused_ops, accept_all_ops)
    checkResult(torch_res, glow_res, atol, rtol)


def scriptVsGlow(
    f,
    atol,
    rtol,
    *inputs,
    expected_fused_ops=None,
    accept_all_ops=False,
    black_list=None,
    use_fp16=False,
    backend_name=None,
):
    if black_list is None:
        black_list = []
    with torch.no_grad():

        torch_res = f(*inputs)

        torch_glow.enableFusionPass()
        torch_glow.setFusionBlacklist(black_list)

        if use_fp16:
            torch_glow.enable_convert_to_fp16()
            torch_glow.enable_convert_fused_to_fp16()
            torch_glow.enable_clip_fp16()
        else:
            torch_glow.disable_convert_to_fp16()
            torch_glow.disable_convert_fused_to_fp16()
            torch_glow.disable_clip_fp16()

        if backend_name:
            torch_glow.setGlowBackend(backend_name)
        else:
            torch_glow.setGlowBackend("Interpreter")

        glow_trace = torch.jit.script(f)
        glow_res = glow_trace(*inputs)

        glow_graph = glow_trace.graph_for(*inputs)
        print("glow_graph,", glow_graph)

        # need to explicitly clear settings to avoid carry-over static settings
        torch_glow.disableFusionPass()
        torch_glow.disable_convert_to_fp16()
        torch_glow.disable_convert_fused_to_fp16()
        torch_glow.disable_clip_fp16()
        torch_glow.setGlowBackend("Interpreter")

    checkExpectedOps(glow_graph, expected_fused_ops, accept_all_ops)
    checkResult(torch_res, glow_res, atol, rtol)


def graph_contains_str(graph, substr):
    return graph.str().find(substr) >= 0
