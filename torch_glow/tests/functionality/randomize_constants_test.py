# isort:skip_file

# pyre-ignore-all-errors
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow
from glow.glow.torch_glow.tests.tests import utils


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(5, 10)

    def forward(self, x):
        return self.linear(x)


def run_model(m, input, randomize):
    torch_glow.disableFusionPass()
    traced_m = torch.jit.trace(m, input)

    if randomize:
        torch_glow.enable_randomize_constants()
    else:
        torch_glow.disable_randomize_constants()

    spec = torch_glow.CompilationSpec()
    spec.get_settings().set_glow_backend("Interpreter")

    compilation_group = torch_glow.CompilationGroup()
    spec.compilation_groups_append(compilation_group)

    input_spec = torch_glow.InputSpec()
    input_spec.set_same_as(input)

    compilation_group.input_sets_append([input_spec])

    glow_m = torch_glow.to_glow(traced_m, {"forward": spec})
    return glow_m(input)


class TestRandomizeWeights(utils.TorchGlowTestCase):
    def test_randomize_weights(self):
        m = Model()
        input = torch.randn(5)
        normal1 = run_model(m, input, False)
        normal2 = run_model(m, input, False)
        rand = run_model(m, input, True)

        assert torch.allclose(normal1, normal2)
        assert not torch.allclose(normal1, rand)
