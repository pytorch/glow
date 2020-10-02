import collections
import copy

import torch


__all__ = [
    "to_glow",
    "to_glow_selective",
    "GlowCompileSpec",
    "InputMeta",
    "CompilationOptions",
]


"""
    InputMeta  C++ custom class that defines metadata of input tensors to
    the module being passed in to_glow. This metadata is composed of
    dimensions and type.
    Usage:
    input_meta = InputMeta()
    input_meta.set(dims, torch.float32)
    inputs = [input_meta, input_meta]
"""
InputMeta = torch.classes.glow.SpecInputMeta


class CompilationOptions:
    r"""
    CompilationOptions is a wrapper around a corresponding C++ custom class.
    It enables to get and set compilation options available in Glow.
    Usage:
    options = GlowOptions()
    options.backend = "Interpreter"
    """

    def __init__(self):
        self.options = torch.classes.glow.PyTorchLoaderSettings()

    @property
    def backend(self):
        return self.options.get_backend_name()

    @backend.setter
    def backend(self, value):
        self.options.set_backend_name(value)

    @property
    def convert_to_fp16(self):
        return self.options.get_convert_to_fp16()

    @convert_to_fp16.setter
    def convert_to_fp16(self, value):
        self.options.set_convert_to_fp16(value)

    @property
    def convert_fused_to_fp16(self):
        return self.options.get_convert_fused_to_fp16()

    @convert_fused_to_fp16.setter
    def convert_fused_to_fp16(self, value):
        self.options.set_convert_fused_to_fp16(value)

    @property
    def saturate_host(self):
        return self.options.get_saturate_host()

    @saturate_host.setter
    def saturate_host(self, value):
        self.options.set_saturate_host(value)

    @property
    def randomize_constants(self):
        return self.options.get_randomize_constants()

    @randomize_constants.setter
    def randomize_constants(self, value):
        self.options.set_randomize_constants(value)

    @property
    def replication_count(self):
        return self.options.get_replication_count()

    @replication_count.setter
    def replication_count(self, value):
        self.options.set_replication_count(value)


class GlowCompileSpec:
    r"""
    GlowCompileSpec is a wrapper around a corresponding C++ custom class.
    The spec contains metadata for the module's input tensor[s] and optional
    compilation options.
    Usage:
    spec = GlowCompileSpec()
    spec.set(inputs, options)
    """

    def __init__(self):
        self.spec = torch.classes.glow.GlowCompileSpec()

    def set(self, input_meta, options=None):
        self.spec.addInputs(input_meta)
        if options is not None:
            self.spec.set_settings(options.options)


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
                method_compile_spec[k] = [v.spec]
            else:
                specs_list = [wrapper.spec for wrapper in v]
                method_compile_spec[k] = specs_list
    elif isinstance(method_compile_spec, list):
        specs_list = [wrapper.spec for wrapper in method_compile_spec]
        method_compile_spec = {"forward": specs_list}
    else:
        method_compile_spec = {"forward": [method_compile_spec.spec]}

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
            info = per_module_info["forward"]
            if not isinstance(info, list):
                info = [info]
        elif isinstance(per_module_info, tuple):
            info = [per_module_info]
        elif isinstance(per_module_info, list):
            info = per_module_info
        else:
            raise ValueError(
                """For each submodule, to_glow_selective expects \
                             either a dictionary of method name -> \
                             (GlowCompileSpec, example_inputs) or just
                             (GlowCompileSpec, example_inputs) and 'forward' \
                             method is assumed"""
            )
        submod = get_submodule(model, path)
        spec_list = []
        for info_tup in info:
            (spec, example_inputs) = info_tup
            spec_list.append(spec)
        submod = torch.jit.trace(submod, example_inputs)
        submod = to_glow(submod, {"forward": spec_list})
        set_submodule(model, path, submod)
    return model
