# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import os
from contextlib import contextmanager
from copy import deepcopy

import torch
import torch_glow

GLOW_FUSION_GROUP = "glow::FusionGroup"
SUBGRAPH_ATTR = "Subgraph"
BACKEND_NAME_KEY = "BACKEND_NAME"
INTERPRETER = "Interpreter"
DEFAULT_BACKEND = os.environ.get(BACKEND_NAME_KEY, "Interpreter")


def get_backend_name():
    return os.environ.get(BACKEND_NAME_KEY, INTERPRETER)


@contextmanager
def ephemeral_torchglow_settings(
    fp16=False, backend=DEFAULT_BACKEND, fusion=False, blocklist=None
):
    old_fp16 = torch_glow.get_convert_to_fp16()
    old_clip = torch_glow.get_clip_fp16()
    old_convert_fused = torch_glow.get_convert_fused_to_fp16()
    old_backend = torch_glow.getGlowBackendName()
    old_blocklist = torch_glow.getFusionBlacklist()
    old_fusion = torch_glow.getFusionPassEnabled()
    try:
        if fusion:
            torch_glow.enableFusionPass()
        else:
            torch_glow.disableFusionPass()
        if fp16:
            torch_glow.enable_convert_to_fp16()
            torch_glow.enable_convert_fused_to_fp16()
            torch_glow.enable_clip_fp16()
        else:
            torch_glow.disable_convert_to_fp16()
            torch_glow.disable_convert_fused_to_fp16()
            torch_glow.disable_clip_fp16()
        if blocklist is None:
            torch_glow.clearFusionBlacklist()
        else:
            torch_glow.setFusionBlacklist(list(blocklist))
        torch_glow.setGlowBackend(backend)
        yield
    finally:
        torch_glow.enable_convert_to_fp16() if old_fp16 else torch_glow.disable_convert_to_fp16()
        torch_glow.enable_clip_fp16() if old_clip else torch_glow.disable_clip_fp16()
        torch_glow.enable_convert_fused_to_fp16() if old_convert_fused else torch_glow.disable_convert_fused_to_fp16()
        torch_glow.enableFusionPass() if old_fusion else torch_glow.disableFusionPass()
        torch_glow.setGlowBackend(old_backend)
        torch_glow.setFusionBlacklist(old_blocklist)


def check_skip(case):
    backend = DEFAULT_BACKEND
    supported = {INTERPRETER}
    try:
        supported = supported | case.supported_backends
    except AttributeError:
        pass

    if backend not in supported:
        case.skipTest("Skipping tests for backend: " + backend)


def generate_glow_spec(module, backend, *inputs):
    spec = torch_glow.CompilationSpec()
    spec.get_settings().set_glow_backend(backend)
    compilation_group = torch_glow.CompilationGroup()
    spec.compilation_groups_append(compilation_group)

    input_specs = []
    for input in inputs:
        input_spec = torch_glow.InputSpec()
        input_spec.set_same_as(input)
        input_specs.append(input_spec)
    compilation_group.input_sets_append(input_specs)
    return spec


def assert_equivalent(result, other_result, atol=5e-4, rtol=1e-3):
    if isinstance(result, tuple) or isinstance(other_result, tuple):
        assert isinstance(result, tuple) and isinstance(other_result, tuple)
        assert len(result) == len(other_result)
        return all(
            assert_equivalent(a, b, atol=atol, rtol=rtol)
            for a, b in zip(result, other_result)
        )
    elif other_result.dtype == torch.bool:
        diff = torch.eq(result, other_result)
        if torch.all(diff):
            return True
        else:
            error = f"Diff:{diff}\n"
            raise AssertionError(error)
    else:
        if torch.allclose(result, other_result, atol, rtol):
            return True
        else:
            diff = torch.abs(result - other_result)
            error = f"First result:\n{result}\n"
            error += f"Second result:\n{other_result}\n"
            error += f"Diff:\n{diff}\n"
            error += f"Max diff:\n{torch.max(diff)}"
            raise AssertionError(error)


def compare_tracing_methods(
    module,
    *inputs,
    atol=5e-4,
    rtol=1e-3,
    reference=None,
    fusible_ops=None,
    fusion_blocklist=None,
    fp16=False,
    scripted=False,
    check_trace=True,
    skip_to_glow=False,  # Ugly hack, TODO: Remove
):
    if not isinstance(module, torch.nn.Module):
        raise AssertionError("to_glow only supports nn.Modules")

    def trace(mod, ins):
        if scripted:
            return torch.jit.script(mod)
        else:
            return torch.jit.trace(mod, ins, check_trace=check_trace)

    with torch.no_grad():
        with ephemeral_torchglow_settings(
            fusion=True, fp16=fp16, blocklist=fusion_blocklist
        ):
            fusion_inputs = deepcopy(inputs)
            fusion_trace = trace(module, fusion_inputs)
            assert_fused(
                fusion_trace.graph_for(*fusion_inputs),
                *(fusible_ops or []),
                accept_any=fusible_ops is None,
            )
            fusion_result = fusion_trace(*fusion_inputs)
        with ephemeral_torchglow_settings(fusion=False, fp16=fp16):
            if scripted:
                torchscript_result = module(*deepcopy(inputs))
            else:
                torchscript_inputs = deepcopy(inputs)
                torchscript_trace = trace(module, torchscript_inputs)
                torchscript_result = torchscript_trace(*torchscript_inputs)
        with ephemeral_torchglow_settings(fusion=False, fp16=fp16):
            if not skip_to_glow:
                glow_inputs = deepcopy(inputs)
                glow_spec = generate_glow_spec(module, DEFAULT_BACKEND, *glow_inputs)
                glow_trace = torch_glow.to_glow(trace(module, glow_inputs), glow_spec)
                glow_result = glow_trace(*glow_inputs)
        if reference:
            assert_equivalent(reference, fusion_trace, atol=atol, rtol=rtol)
            assert_equivalent(reference, torchscript_result, atol=atol, rtol=rtol)
            if not skip_to_glow:
                assert_equivalent(reference, glow_result, atol=atol, rtol=rtol)
        # This is written out manually instead of using combinations in order to aid
        # debugging. TODO: Clean up.
        assert_equivalent(fusion_result, torchscript_result, atol=atol, rtol=rtol)
        if not skip_to_glow:
            assert_equivalent(fusion_result, glow_result, atol=atol, rtol=rtol)
            assert_equivalent(torchscript_result, glow_result, atol=atol, rtol=rtol)


def assert_fused(fused_graph, *ops, accept_any=False, strict=False):
    expected = set(ops)
    fused = set()
    with torch.no_grad():
        for node in fused_graph.nodes():
            kind = node.kind()
            if kind == GLOW_FUSION_GROUP:
                fused.update(map(lambda n: n.kind(), node.g(SUBGRAPH_ATTR).nodes()))
            else:
                assert kind not in expected, f"Expected {kind} to be fused"
    missing = set() if (accept_any and fused) else expected - fused
    unexpected = set() if (accept_any or not strict) else fused - expected
    assert not unexpected, f"Expected fusion of {expected}, but {fused} was fused."
    assert not missing, f"Expected fusion of {expected}, but only {fused} was fused."


def graph_contains_str(graph, substr):
    return graph.str().find(substr) >= 0


# Verifies equal modules for save-load tests.
def assertModulesEqual(case, mod1, mod2, message=None):
    for p1, p2 in itertools.zip_longest(mod1.parameters(), mod2.parameters()):
        case.assertTrue(p1.equal(p2), message)
