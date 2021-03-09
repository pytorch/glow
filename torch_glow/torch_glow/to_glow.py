import collections
import copy
from typing import List, Tuple, Any

import torch


__all__ = [
    "to_glow",
    "to_glow_selective",
    "get_submod_inputs",
    "CompilationSpec",
    "CompilationGroup",
    "InputSpec",
    "CompilationSpecSettings",
    "FuserSettings",
    "input_spec_from_tensor",
    "input_specs_from_tensors",
    "generate_glow_compilation_spec",
]

CompilationSpec = torch.classes.glow.CompilationSpec
CompilationGroup = torch.classes.glow.CompilationGroup
InputSpec = torch.classes.glow.InputSpec
CompilationSpecSettings = torch.classes.glow.CompilationSpecSettings
FuserSettings = torch.classes.glow.FuserSettings


def input_spec_from_tensor(tensor: torch.Tensor) -> InputSpec:
    input_spec = InputSpec()
    input_spec.set_same_as(tensor)
    return input_spec


def input_specs_from_tensors(tensors: List[torch.Tensor]) -> List[InputSpec]:
    return [input_spec_from_tensor(tensor) for tensor in tensors]


def generate_glow_compilation_spec(model, backend, *example_inputs):
    spec = CompilationSpec()
    spec.get_settings().set_glow_backend(backend)
    compilation_group = CompilationGroup()
    compilation_group.input_sets_append(input_specs_from_tensors(example_inputs))
    spec.compilation_groups_append(compilation_group)
    return spec


def to_glow(model, method_compile_spec):
    r"""Lower a model to Glow

    to_glow is a wrapper around the torch._C._jit_to_backend which lowers the
    the specified module `mod` to Glow using the the MethodCompileSpec
    `method_compile_spec`. MethodCompileSpec is a dictionary from method name
    in `mod` such as 'forward' to CompilationSpec for that method

    Args:
        model: Model to be lowered to glow
        method_compile_spec: Either a dicionary from method name to
                           CompilationSpec or just a CompilationSpec and method
                           name is assumed to be "forward"

    Return:
        A copy of the model that has been lowered to Glow and will run on
        Glow backend devices
    """
    if not isinstance(method_compile_spec, collections.Mapping):
        method_compile_spec = {"forward": method_compile_spec}

    return torch._C._jit_to_backend("glow", model, method_compile_spec)


def check_module_names(module_names):
    """Checks that module names don't overlap at all"""
    assert "" not in module_names, "Use to_glow to lower top level module"
    for path1 in module_names:
        for path2 in module_names:
            if path1 == path2:
                continue
            assert (
                path1 not in path2
            ), f"Can't to_glow a module nested inside another to_glow module, \
                    found {path2} inside of {path1}"


def get_submodule(mod, path):
    path = path.split(".")
    assert len(path) > 0
    found_mod = mod
    for item in path:
        found_mod = getattr(found_mod, item)
    return found_mod


def set_submodule(mod, path, submod):
    path = path.split(".")
    assert len(path) > 0
    found_mod = mod
    for item in path[:-1]:
        found_mod = getattr(found_mod, item)
    setattr(found_mod, path[-1], submod)
    pass


def get_submod_inputs(
    mod: torch.nn.Module, path: str, example_inputs: Any
) -> Tuple[torch.Tensor]:
    r"""Get the inputs of a submodule given the top-level model
    and its input.

    Register a forward hook that record the inputs of the submodule
    and then run the model to triger the hook.

    Args:
        mod: top-level model
        path: path to a submodule
        example_inputs: inputs to the top-level model

    Return:
        inputs: Tuple[torch.Tensor]
    """
    submod = get_submodule(mod, path)
    sub_inputs = None

    def get_inputs(self: torch.nn.Module, inputs: Any):
        nonlocal sub_inputs
        sub_inputs = inputs

    handle = submod.register_forward_pre_hook(get_inputs)
    mod(*example_inputs)
    handle.remove()
    return sub_inputs


def to_glow_selective(model, specs_and_examples, inplace=False):
    r"""Selectively lowers submodules of the given module to Glow.

    Instead of using to_glow to lower an entire module to Glow,
    to_glow_selective can be used to selectively find and replace submodules in
    the given module with a version of the module that is traced and lowered
    to Glow. Each specified submodule is lowered independently and so will be
    a separate compilation unit in Glow.

    Args:
        model: top-level model to be selectively lowered
        specs_and_examples: A dictionary with keys that name submodules
                           recursively from model and values that are the a
                           tuple of (CompilationSpec, example_inputs) where
                           example_inputs are inputs that are used to trace
                           the submodule.
        inplace: Carry out model transformations in-place, the original module
                is mutated

    Return:
        Model with selectively lowered submodules
    """
    check_module_names(list(specs_and_examples.keys()))
    if not inplace:
        model = copy.deepcopy(model)
    if isinstance(model, torch.jit._script.RecursiveScriptModule):
        spec_list, path_list = [], []
        submod_idx = 0
        for path, spec in specs_and_examples.items():
            spec_list.append(spec)
            path_list.append(path)

        def to_glow_helper(submod):
            nonlocal submod_idx
            res_model = to_glow(submod, {"forward": spec_list[submod_idx]})
            submod_idx += 1
            return res_model

        model = torch._C._jit_to_backend_selective(model, to_glow_helper, path_list)
    else:
        for path, (spec, example_inputs) in specs_and_examples.items():
            submod = get_submodule(model, path)
            submod = torch.jit.trace(submod, example_inputs)
            submod = to_glow(submod, {"forward": spec})
            set_submodule(model, path, submod)
    return model
