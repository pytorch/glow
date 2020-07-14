import torch

__all__ = ["to_glow"]


def to_glow(mod, method_compile_spec):
    """to_glow is a wrapper around the torch._C._jit_to_backend which lowers the
       the specified module `mod` to Glow using the the MethodCompileSpec
       `method_compile_spec`. MethodCompileSpec is a dictionary from method name
       in `mod` such as 'forward' to GlowCompileSpec for that method."""
    return torch._C._jit_to_backend("glow", mod._c, method_compile_spec)
