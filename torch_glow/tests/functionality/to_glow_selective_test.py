# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch_glow


class Qux(torch.nn.Module):
    def __init__(self, x):
        super(Qux, self).__init__()
        self.x = x

    def forward(self, a, b):
        return a - b - self.x


class Baz(torch.nn.Module):
    def __init__(self, x):
        super(Baz, self).__init__()
        self.x = x

    def forward(self, a, b):
        return a + b * self.x


class Bar(torch.nn.Module):
    def __init__(self, x):
        super(Bar, self).__init__()
        self.x = x

    def forward(self, a, b):
        return a * b + self.x


class Foo(torch.nn.Module):
    def __init__(self, bar, baz):
        super(Foo, self).__init__()
        self.bar = bar
        self.baz = baz

    def forward(self, a, b):
        return self.baz(self.bar(a, b), b)


class Model(torch.nn.Module):
    def __init__(self, foo, qux):
        super(Model, self).__init__()
        self.foo = foo
        self.qux = qux

    def forward(self, a, b):
        return self.qux(self.foo(a, b), a)


r"""
            model
            /   \
          foo    qux (Glow)
        /    \
  bar (Glow)  baz
"""

bar = Bar(4.0)
baz = Baz(2.0)
qux = Qux(3.0)
foo = Foo(bar, baz)
model = Model(foo, qux)


class TestSelectiveToGlow(unittest.TestCase):
    def test_to_glow_selective(self):
        a = torch.zeros(4) + 8
        b = torch.zeros(4) + 7
        torch_res = model(a, b)

        spec = torch_glow.CompilationSpec()
        spec.get_settings().set_glow_backend("Interpreter")

        compilation_group = torch_glow.CompilationGroup()
        spec.compilation_groups_append(compilation_group)

        a_spec = torch_glow.InputSpec()
        a_spec.set_same_as(a)
        b_spec = torch_glow.InputSpec()
        b_spec.set_same_as(b)

        compilation_group.input_sets_append([a_spec, b_spec])

        glow_mod = torch_glow.to_glow_selective(
            model, {"foo.bar": (spec, (a, b)), "qux": (spec, (a, b))}
        )

        glow_mod = torch.jit.trace(glow_mod, (a, b))
        glow_res = glow_mod(a, b)

        assert torch.allclose(torch_res, glow_res)
