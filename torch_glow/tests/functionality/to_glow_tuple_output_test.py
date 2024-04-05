# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import io

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils
from glow.glow.torch_glow.tests.tests.utils import assertModulesEqual


class TwoTupleModule(torch.nn.Module):
    def __init__(self):
        super(TwoTupleModule, self).__init__()

    def forward(self, x):
        y = 2 * x
        return (x, y)


class OneTupleModule(torch.nn.Module):
    def __init__(self):
        super(OneTupleModule, self).__init__()

    def forward(self, x):
        y = 2 * x
        return (y,)


class TestToGlowTupleOutput(utils.TorchGlowTestCase):
    def tuple_test_helper(self, ModType):
        input = torch.randn(4)

        model = ModType()

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
        g = lowered_model(input)

        # Run reference model
        t = model(input)

        self.assertEqual(type(g), type(t))
        self.assertEqual(len(g), len(t))

        for gi, ti in zip(g, t):
            self.assertTrue(torch.allclose(gi, ti))

        # test module ser/de with tuple output
        buffer = io.BytesIO()
        torch.jit.save(lowered_model, buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)
        assertModulesEqual(self, lowered_model, loaded_model)

    def test_to_glow_one_tuple_output(self):
        self.tuple_test_helper(OneTupleModule)

    def test_to_glow_two_tuple_output(self):
        self.tuple_test_helper(TwoTupleModule)
