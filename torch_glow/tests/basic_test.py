import torch
import torch_glow

@torch.jit.script
def foo(a, b):
  c = a.mul(b)
  a = c.mul(c)
  a = c.mul(a)
  d = c.div(a)
  return d

torch_glow.enableFusionPass()

@torch.jit.script
def foo_glow(a, b):
    return foo(a, b)

def test_foo():
  x = torch.randn(4)
  y = torch.randn(4)

  jit_res = foo(x, y)
  jit_glow_res = foo_glow(x, y)

  assert torch.allclose(jit_res, jit_glow_res)
  