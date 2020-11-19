# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow

x = torch.randn(4)
y = torch.randn(4)


@torch.jit.script
def foo(a, b):
    c = a.mul(b)
    a = c.mul(c)
    a = c.mul(a)
    d = c.div(a)
    return d


print("original jit ir")
print(foo.graph_for(x, y))

jit_res = foo(x, y)

torch_glow.enableFusionPass()


@torch.jit.script
def foo_glow(a, b):
    return foo(a, b)


print("glow jit ir")
print(foo_glow.graph_for(x, y))

jit_glow_res = foo_glow(x, y)

print("jit_res")
print(jit_res)
print("jit_glow_res")
print(jit_glow_res)

assert torch.allclose(jit_res, jit_glow_res)
