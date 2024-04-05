# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils


class Foo(torch.nn.Module):
    def __init__(self):
        super(Foo, self).__init__()

    def forward(self, x, y):
        return x + y


class TestToGlowMultpleInputSets(utils.TorchGlowTestCase):
    def test_to_glow_multiple_groups_and_input_sets(self):
        x1 = torch.randn(1, 4)
        y1 = torch.randn(2, 4)

        x2 = torch.randn(1, 2)
        y2 = torch.randn(5, 2)

        x3 = torch.randn(7)
        y3 = torch.randn(3, 7)

        mod = Foo()
        scripted_mod = torch.jit.script(mod)

        x1_y1_set = torch_glow.input_specs_from_tensors([x1, y1])
        x2_y2_set = torch_glow.input_specs_from_tensors([x2, y2])
        x3_y3_set = torch_glow.input_specs_from_tensors([x3, y3])

        # Create two CompilationGroup, first one contains two input sets
        # and the second CompilationGroup has the third input set
        spec = torch_glow.CompilationSpec()
        spec.get_settings().set_glow_backend("Interpreter")

        compilation_group_1 = torch_glow.CompilationGroup()
        compilation_group_2 = torch_glow.CompilationGroup()
        spec.compilation_groups_append(compilation_group_1)
        spec.compilation_groups_append(compilation_group_2)

        compilation_group_1.input_sets_append(x1_y1_set)
        compilation_group_1.input_sets_append(x2_y2_set)
        compilation_group_2.input_sets_append(x3_y3_set)

        lowered_module = torch_glow.to_glow(scripted_mod, spec)

        torch_res1 = mod(x1, y1)
        torch_res2 = mod(x2, y2)
        torch_res3 = mod(x3, y3)

        glow_res1 = lowered_module(x1, y1)
        glow_res2 = lowered_module(x2, y2)
        glow_res3 = lowered_module(x3, y3)

        assert torch.allclose(torch_res1, glow_res1)
        assert torch.allclose(torch_res2, glow_res2)
        assert torch.allclose(torch_res3, glow_res3)
