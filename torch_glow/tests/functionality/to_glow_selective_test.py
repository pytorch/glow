# isort:skip_file

# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils


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
        return self.baz(self.bar(a.reshape(1, -1), b.reshape(1, -1)), b)


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


def get_compilation_spec(inputs):
    """helper function to get the compilation spec of the submodule"""
    spec = torch_glow.CompilationSpec()
    spec.get_settings().set_glow_backend("Interpreter")

    compilation_group = torch_glow.CompilationGroup()
    spec.compilation_groups_append(compilation_group)

    compilation_group.input_sets_append(torch_glow.input_specs_from_tensors(inputs))
    return spec


class TestSelectiveToGlow(utils.TorchGlowTestCase):
    def test_to_glow_selective(self):
        inputs = (torch.zeros(4) + 8, torch.zeros(4) + 7)
        torch_res = model(*inputs)

        bar_inputs = torch_glow.get_submod_inputs(model, "foo.bar", inputs)
        qux_inputs = torch_glow.get_submod_inputs(model, "qux", inputs)

        glow_mod = torch_glow.to_glow_selective(
            model,
            {
                "foo.bar": (get_compilation_spec(bar_inputs), bar_inputs),
                "qux": (get_compilation_spec(qux_inputs), qux_inputs),
            },
            inplace=False,
        )

        glow_mod = torch.jit.trace(glow_mod, inputs)
        glow_res = glow_mod(*inputs)

        assert torch.allclose(torch_res, glow_res)

    def test_to_glow_selective_already_scripted(self):
        inputs = (torch.zeros(4) + 8, torch.zeros(4) + 7)
        torch_res = model(*inputs)

        bar_inputs = torch_glow.get_submod_inputs(model, "foo.bar", inputs)
        qux_inputs = torch_glow.get_submod_inputs(model, "qux", inputs)

        with torch.no_grad():
            traced_model = torch.jit.trace(model, inputs)

        glow_mod = torch_glow.to_glow_selective(
            traced_model,
            {
                "foo.bar": get_compilation_spec(bar_inputs),
                "qux": get_compilation_spec(qux_inputs),
            },
            inplace=False,
        )
        glow_res = glow_mod(*inputs)
        assert torch.allclose(torch_res, glow_res)
