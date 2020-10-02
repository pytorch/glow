from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch_glow


class Foo(torch.nn.Module):
    def __init__(self):
        super(Foo, self).__init__()

    def forward(self, x):
        y = 2 * x
        return x, y


class TestToGlowTupleOutput(unittest.TestCase):
    def test_to_glow_tuple_output(self):
        a = torch.randn(4)

        model = Foo()
        torch_resA = model(a)
        (tx, ty) = torch_resA

        metaA = torch_glow.InputMeta()
        metaA.set_same_as(a)
        inputA = [metaA]

        options = torch_glow.CompilationOptions()
        options.backend = "Interpreter"
        specA = torch_glow.GlowCompileSpec()
        specA.set(inputA, options)

        scripted_mod = torch.jit.script(model)
        lowered_mod = torch_glow.to_glow(scripted_mod, [specA])
        glow_resA = lowered_mod(a)
        (gx, gy) = glow_resA

        assert torch.allclose(tx, gx)
        assert torch.allclose(ty, gy)
