import collections
import copy

import torch


__all__ = ["to_glow", "to_glow_selective"]


def to_glow(model, method_compile_spec):
    r"""Lower a model to Glow

    to_glow is a wrapper around the torch._C._jit_to_backend which lowers the
    the specified module `mod` to Glow using the the MethodCompileSpec
    `method_compile_spec`. MethodCompileSpec is a dictionary from method name
    in `mod` such as 'forward' to GlowCompileSpec for that method

    Args:
        model: Model to be lowered to glow
        specs_and_examples: Either a dicionary from method name to
                           GlowCompileSpec or just a GlowCompileSpec and method
                           name is assumed to be "forward"

    Return:
        A copy of the model that has been lowered to Glow and will run on
        Glow backend devices
    """
    if isinstance(method_compile_spec, collections.Mapping):
        for k, v in method_compile_spec.items():
            if not isinstance(v, list):
                method_compile_spec[k] = [v]
    elif isinstance(method_compile_spec, list):
        method_compile_spec = {"forward", method_compile_spec}
    else:
        method_compile_spec = {"forward", [method_compile_spec]}

    return torch._C._jit_to_backend("glow", model._c, method_compile_spec)


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


def to_glow_selective(model, specs_and_examples, inplace=False):
    r"""Selectively lowers submodules of the given module to Glow.

    Instead of using to_glow to lower an entire module to Glow,
    to_glow_selective can be used to selective find and replace submodules in
    the given module with a version of the module that is traced and lowered
    to Glow. Each specified submodule is lowered independently and so will be
    a separate compilation unit in Glow.

    Args:
        model: top-level model to be selectively lowered
        specs_and_examples: A dictionary with keys that name submodules
                           recursively from model and values that are either
                           dicionaries from method name to tuple of
                           GlowCompileSpec used for calling to_glow and example
                           inputs used for tracing or just that tuple without
                           the method name and method name is assumed to be
                           "forward"
        inplace: Carry out model transformations in-place, the original module
                is mutated

    Return:
        Model with selectively lowered submodules
    """
    check_module_names(list(specs_and_examples.keys()))
    if not inplace:
        model = copy.deepcopy(model)
    for path, per_module_info in specs_and_examples.items():
        if isinstance(per_module_info, collections.Mapping):
            assert (
                len(per_module_info) == 1 and "forward" in per_module_info
            ), "Only forward method is supported by to_glow_selective for now"
            (spec, example_inputs) = per_module_info["forward"]
        elif isinstance(per_module_info, tuple):
            (spec, example_inputs) = per_module_info
        else:
            raise ValueError(
                """For each submodule, to_glow_selective expects \
                             either a dictionary of method name -> \
                             (GlowCompileSpec, example_inputs) or just
                             (GlowCompileSpec, example_inputs) and 'forward' \
                             method is assumed"""
            )
        submod = get_submodule(model, path)
        submod = torch.jit.trace(submod, example_inputs)
        submod = to_glow(submod, {"forward": spec})
        set_submodule(model, path, submod)
    return model
