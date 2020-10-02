from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch_glow


class Foo(torch.nn.Module):
    def __init__(self):
        super(Foo, self).__init__()

    def forward(self, x):
        return x + x


class Bar(torch.nn.Module):
    def __init__(self):
        super(Bar, self).__init__()

    def forward(self, x, y):
        return x + y


class Model(torch.nn.Module):
    def __init__(self, foo, bar):
        super(Model, self).__init__()
        self.foo = foo
        self.bar = bar

    def forward(self, x, y):
        return self.foo(x) + self.bar(x, y)


class TestToGlowMultiSpec(unittest.TestCase):
    def test_to_glow_multiple_specs(self):
        a = torch.randn(4)
        b = torch.randn(6)

        model = Foo()
        torch_resA = model(a)
        torch_resB = model(b)

        metaA = torch_glow.InputMeta()
        metaA.set_same_as(a)
        inputA = [metaA]

        metaB = torch_glow.InputMeta()
        metaB.set_same_as(b)
        inputB = [metaB]

        options = torch_glow.CompilationOptions()
        options.backend = "Interpreter"
        specA = torch_glow.GlowCompileSpec()
        specA.set(inputA, options)
        specB = torch_glow.GlowCompileSpec()
        specB.set(inputB, options)

        scripted_mod = torch.jit.script(model)
        lowered_mod = torch_glow.to_glow(scripted_mod, [specA, specB])
        glow_resA = lowered_mod(a)
        glow_resB = lowered_mod(b)

        assert torch.allclose(torch_resA, glow_resA)
        assert torch.allclose(torch_resB, glow_resB)

    def test_to_glow_selective_multi_spec(self):
        a = torch.randn(4)
        b = torch.randn(6)
        foo = Foo()
        bar = Bar()
        model = Model(foo, bar)
        torch_resA = model(a, a)
        torch_resB = model(b, b)

        metaA = torch_glow.InputMeta()
        metaA.set_same_as(a)
        inputA = [metaA]

        metaB = torch_glow.InputMeta()
        metaB.set_same_as(b)
        inputB = [metaB]

        options = torch_glow.CompilationOptions()
        options.backend = "Interpreter"
        specA = torch_glow.GlowCompileSpec()
        specA.set(inputA, options)
        specB = torch_glow.GlowCompileSpec()
        specB.set(inputB, options)

        lowered_mod = torch_glow.to_glow_selective(
            model, {"foo": [(specA, (a)), (specB, (b))]}
        )
        glow_resA = lowered_mod(a, a)
        glow_resB = lowered_mod(b, b)

        assert torch.allclose(torch_resA, glow_resA)
        assert torch.allclose(torch_resB, glow_resB)
