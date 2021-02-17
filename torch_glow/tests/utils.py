# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import os
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO

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


def assert_equivalent(
    result1_name, result1, result2_name, result2, atol=5e-4, rtol=1e-3
):
    if isinstance(result1, tuple) or isinstance(result2, tuple):
        assert isinstance(result1, tuple) and isinstance(result2, tuple)
        assert len(result1) == len(result2)
        return all(
            assert_equivalent(result1_name, a, result2_name, b, atol=atol, rtol=rtol)
            for a, b in zip(result1, result2)
        )
    elif result2.dtype == torch.bool:
        diff = torch.eq(result1, result2)
        if torch.all(diff):
            return True
        else:
            error = f"Diff:{diff}\n"
            raise AssertionError(error)
    else:
        if torch.allclose(result1, result2, atol, rtol):
            return True
        else:
            diff = torch.abs(result1 - result2)
            error = f"{result1_name} result:\n{result1}\n"
            error += f"{result2_name} result:\n{result2}\n"
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
                glow_spec = torch_glow.generate_glow_compilation_spec(
                    module, DEFAULT_BACKEND, *glow_inputs
                )
                glow_trace = torch_glow.to_glow(trace(module, glow_inputs), glow_spec)
                glow_result = glow_trace(*glow_inputs)
        if reference:
            assert_equivalent(
                "Reference",
                reference,
                "Glow fusion",
                fusion_trace,
                atol=atol,
                rtol=rtol,
            )
            assert_equivalent(
                "Reference",
                reference,
                "TorchScript",
                torchscript_result,
                atol=atol,
                rtol=rtol,
            )
            if not skip_to_glow:
                assert_equivalent(
                    "Reference", reference, "Glow", glow_result, atol=atol, rtol=rtol
                )
        # This is written out manually instead of using combinations in order to aid
        # debugging. TODO: Clean up.
        assert_equivalent(
            "Glow fusion",
            fusion_result,
            "TorchScript",
            torchscript_result,
            atol=atol,
            rtol=rtol,
        )
        if not skip_to_glow:
            assert_equivalent(
                "Glow fusion", fusion_result, "Glow", glow_result, atol=atol, rtol=rtol
            )
            assert_equivalent(
                "TorchScript",
                torchscript_result,
                "Glow",
                glow_result,
                atol=atol,
                rtol=rtol,
            )


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


def save_and_reload_model(model):
    buf = BytesIO()

    print("saving ...")
    torch.jit.save(model, buf)
    print("done")

    print("reloading....")
    buf.seek(0)
    reloaded_model = torch.jit.load(buf)
    print("done")
    return reloaded_model
