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
        input = torch.randn(4)

        model = Foo()

        spec = torch_glow.CompilationSpec()
        spec.get_settings().set_glow_backend("Interpreter")

        compilation_group = torch_glow.CompilationGroup()
        spec.compilation_groups_append(compilation_group)

        input_spec = torch_glow.InputSpec()
        input_spec.set_same_as(input)

        compilation_group.input_sets_append([input_spec])

        scripted_mod = torch.jit.script(model)
        lowered_model = torch_glow.to_glow(scripted_mod, {"forward": spec})

        # Run Glow model
        (gx, gy) = lowered_model(input)

        # Run reference model
        (tx, ty) = model(input)

        assert torch.allclose(tx, gx)
        assert torch.allclose(ty, gy)
